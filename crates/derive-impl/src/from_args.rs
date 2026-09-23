use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::ext::IdentExt;
use syn::meta::ParseNestedMeta;
use syn::{Attribute, Data, DeriveInput, Expr, Field, Ident, Lit, Result, Token, parse_quote};

/// The kind of the python parameter, this corresponds to the value of Parameter.kind
/// (https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind)
#[derive(Clone, Copy, Default, Eq, PartialEq)]
enum ParameterKind {
    PositionalOnly,
    #[default]
    PositionalOrKeyword,
    KeywordOnly,
    Flatten,
}

impl TryFrom<&Ident> for ParameterKind {
    type Error = ();

    fn try_from(ident: &Ident) -> core::result::Result<Self, Self::Error> {
        Ok(match ident.to_string().as_str() {
            "positional" => Self::PositionalOnly,
            "any" => Self::PositionalOrKeyword,
            "named" => Self::KeywordOnly,
            "flatten" => Self::Flatten,
            _ => return Err(()),
        })
    }
}

// None == quote!(Default::default())
type DefaultValue = Option<Expr>;

#[derive(Default)]
struct ArgAttribute {
    name: Option<String>,
    kind: ParameterKind,
    default: Option<DefaultValue>,
    error_msg: Option<String>,
}

impl ArgAttribute {
    fn from_attribute(attr: &Attribute) -> Option<Result<Self>> {
        if !attr.path().is_ident("pyarg") {
            return None;
        }

        let inner = move || {
            let mut arg_attr = None;
            attr.parse_nested_meta(|meta| {
                let Some(arg_attr) = &mut arg_attr else {
                    let kind = meta
                        .path
                        .get_ident()
                        .and_then(|ident| ParameterKind::try_from(ident).ok())
                        .ok_or_else(|| {
                            meta.error(
                                "The first argument to #[pyarg()] must be the parameter type, \
                                 either 'positional', 'any', 'named', or 'flatten'.",
                            )
                        })?;
                    arg_attr = Some(Self {
                        name: None,
                        kind,
                        default: None,
                        error_msg: None,
                    });
                    return Ok(());
                };
                arg_attr.parse_argument(meta)
            })?;
            arg_attr
                .ok_or_else(|| err_span!(attr, "There must be at least one argument to #[pyarg()]"))
        };
        Some(inner())
    }

    fn parse_argument(&mut self, meta: ParseNestedMeta<'_>) -> Result<()> {
        if self.kind == ParameterKind::Flatten {
            return Err(meta.error("can't put additional arguments on a flatten arg"));
        }

        if meta.path.is_ident("default") && meta.input.peek(Token![=]) {
            if matches!(self.default, Some(Some(_))) {
                return Err(meta.error("Default already set"));
            }
            let val = meta.value()?;
            self.default = Some(Some(val.parse()?))
        } else if meta.path.is_ident("default") || meta.path.is_ident("optional") {
            if self.default.is_none() {
                self.default = Some(None);
            }
        } else if meta.path.is_ident("name") {
            if self.name.is_some() {
                return Err(meta.error("already have a name"));
            }
            let val = meta.value()?.parse::<syn::LitStr>()?;
            self.name = Some(val.value())
        } else if meta.path.is_ident("error_msg") {
            if self.error_msg.is_some() {
                return Err(meta.error("already have an error_msg"));
            }
            let val = meta.value()?.parse::<syn::LitStr>()?;
            self.error_msg = Some(val.value())
        } else {
            return Err(meta.error("Unrecognized pyarg attribute"));
        }

        Ok(())
    }
}

impl TryFrom<&Field> for ArgAttribute {
    type Error = syn::Error;

    fn try_from(field: &Field) -> core::result::Result<Self, Self::Error> {
        let mut pyarg_attrs = field
            .attrs
            .iter()
            .filter_map(Self::from_attribute)
            .collect::<core::result::Result<Vec<_>, _>>()?;

        if pyarg_attrs.len() >= 2 {
            bail_span!(field, "Multiple pyarg attributes on field")
        };

        Ok(pyarg_attrs.pop().unwrap_or_default())
    }
}

fn generate_field((i, field): (usize, &Field)) -> Result<TokenStream> {
    let attr = ArgAttribute::try_from(field)?;
    let name = field.ident.as_ref();
    let name_string = name.map(|ident| ident.unraw().to_string());
    if matches!(&name_string, Some(s) if s.starts_with("_phantom")) {
        return Ok(quote! {
            #name: ::std::marker::PhantomData,
        });
    }

    let field_name = match name {
        Some(id) => id.to_token_stream(),
        None => syn::Index::from(i).into_token_stream(),
    };

    if let ParameterKind::Flatten = attr.kind {
        return Ok(quote! {
            #field_name: ::rustpython_vm::function::FromArgs::from_args(vm, args)?,
        });
    }

    let pyname = attr
        .name
        .or(name_string)
        .ok_or_else(|| err_span!(field, "field in tuple struct must have name attribute"))?;

    let middle = if let Some(error_msg) = &attr.error_msg {
        quote! {
            .map(|x| ::rustpython_vm::convert::TryFromObject::try_from_object(vm, x)
                .map_err(|_| vm.new_type_error(#error_msg))).transpose()?
        }
    } else {
        quote! {
            .map(|x| ::rustpython_vm::convert::TryFromObject::try_from_object(vm, x)).transpose()?
        }
    };

    let ending = if let Some(default) = attr.default {
        let ty = &field.ty;
        let default = default.unwrap_or_else(|| parse_quote!(::std::default::Default::default()));
        quote! {
            .map(<#ty as ::rustpython_vm::function::FromArgOptional>::from_inner)
            .unwrap_or_else(|| #default)
        }
    } else {
        // A parameter a call may name is named back when it is missing; one
        // that can only be passed by position is only ever counted.
        let err = match attr.kind {
            ParameterKind::PositionalOnly => quote! {
                ::rustpython_vm::function::ArgumentError::TooFewArgs
            },
            ParameterKind::PositionalOrKeyword | ParameterKind::KeywordOnly => {
                let pos = i + 1;
                quote! {
                    ::rustpython_vm::function::ArgumentError::MissingRequiredArgument {
                        name: #pyname.to_owned(),
                        pos: #pos,
                    }
                }
            }
            ParameterKind::Flatten => unreachable!(),
        };
        quote! {
            .ok_or_else(|| #err)?
        }
    };

    let file_output = match attr.kind {
        ParameterKind::PositionalOnly => quote! {
            #field_name: args.take_positional()#middle #ending,
        },
        ParameterKind::PositionalOrKeyword => quote! {
            #field_name: args.take_positional_keyword(#pyname)#middle #ending,
        },
        ParameterKind::KeywordOnly => quote! {
            #field_name: args.take_keyword(#pyname)#middle #ending,
        },
        ParameterKind::Flatten => unreachable!(),
    };

    Ok(file_output)
}

fn compute_arity_bounds(field_attrs: &[ArgAttribute]) -> (usize, usize) {
    let positional_fields = field_attrs.iter().filter(|attr| {
        matches!(
            attr.kind,
            ParameterKind::PositionalOnly | ParameterKind::PositionalOrKeyword
        )
    });

    let min_arity = positional_fields
        .clone()
        .filter(|attr| attr.default.is_none())
        .count();
    let max_arity = positional_fields.count();

    (min_arity, max_arity)
}

fn python_default_repr(default: Option<&DefaultValue>) -> Option<String> {
    match default {
        None => None,
        Some(None) => Some("None".to_owned()),
        Some(Some(expr)) => match expr {
            Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
                Lit::Bool(b) => Some(if b.value {
                    "True".to_owned()
                } else {
                    "False".to_owned()
                }),
                Lit::Int(i) => Some(i.base10_digits().to_owned()),
                Lit::Float(f) => Some(f.base10_digits().to_owned()),
                Lit::Str(s) => Some(format!("'{}'", s.value())),
                _ => Some("None".to_owned()),
            },
            _ => Some("None".to_owned()),
        },
    }
}

pub(crate) fn impl_from_args(input: DeriveInput) -> Result<TokenStream> {
    let struct_fields = match input.data {
        Data::Struct(syn::DataStruct { fields, .. }) => fields,
        _ => bail_span!(input, "FromArgs input must be a struct"),
    };

    let generated_fields = struct_fields
        .iter()
        .enumerate()
        .map(generate_field)
        .collect::<Result<TokenStream>>()?;
    let field_attrs = struct_fields
        .iter()
        .filter_map(|field| field.try_into().ok())
        .collect::<Vec<ArgAttribute>>();

    let (min_arity, max_arity) = compute_arity_bounds(&field_attrs);

    let takes_keywords_named = field_attrs.iter().any(|attr| {
        matches!(
            attr.kind,
            ParameterKind::KeywordOnly | ParameterKind::PositionalOrKeyword
        )
    });
    let flatten_tys = struct_fields
        .iter()
        .zip(&field_attrs)
        .filter(|(_, attr)| attr.kind == ParameterKind::Flatten)
        .map(|(field, _)| field.ty.clone())
        .collect::<Vec<_>>();
    let takes_keywords = quote! {
        #takes_keywords_named #(|| <#flatten_tys as ::rustpython_vm::function::FromArgs>::TAKES_KEYWORDS)*
    };
    let variable_arity = {
        let from_bounds = min_arity != max_arity;
        quote! {
            #from_bounds #(|| <#flatten_tys as ::rustpython_vm::function::FromArgs>::VARIABLE_ARITY)*
        }
    };

    let all_keyword_only = !field_attrs.is_empty()
        && field_attrs
            .iter()
            .all(|attr| attr.kind == ParameterKind::KeywordOnly);
    let keyword_only_signature = if all_keyword_only {
        let mut parts = Vec::new();
        for (field, attr) in struct_fields.iter().zip(&field_attrs) {
            let Some(name) = attr
                .name
                .clone()
                .or_else(|| field.ident.as_ref().map(|ident| ident.unraw().to_string()))
            else {
                continue;
            };
            if let Some(default) = python_default_repr(attr.default.as_ref()) {
                parts.push(format!("{name}={default}"));
            } else {
                parts.push(name);
            }
        }
        let joined = parts.join(", ");
        quote!(Some(#joined))
    } else {
        quote!(None)
    };

    let max_positional = if all_keyword_only { 0 } else { max_arity };

    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let output = quote! {
        impl #impl_generics ::rustpython_vm::function::FromArgs for #name #ty_generics #where_clause {
            const TAKES_KEYWORDS: bool = #takes_keywords;
            const VARIABLE_ARITY: bool = #variable_arity;
            const MAX_POSITIONAL: usize = #max_positional;
            const KEYWORD_ONLY_SIGNATURE: Option<&'static str> = #keyword_only_signature;

            fn arity() -> ::std::ops::RangeInclusive<usize> {
                #min_arity..=#max_arity
            }

            fn from_args(
                vm: &::rustpython_vm::VirtualMachine,
                args: &mut ::rustpython_vm::function::FuncArgs
            ) -> ::core::result::Result<Self, ::rustpython_vm::function::ArgumentError> {
                Ok(Self { #generated_fields })
            }
        }
    };
    Ok(output)
}
