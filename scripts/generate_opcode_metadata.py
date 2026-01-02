"""Generate _opcode_metadata.py for RustPython bytecode.

This file generates opcode metadata that is compatible with both:
1. dis.py (which uses the opmap to understand bytecode)
2. _opcode module functions (has_const, has_name, etc.)

The key insight is that RustPython uses its own Instruction enum with sequential
numbering (0, 1, 2, ...) instead of CPython's opcode numbers.
"""
import re

# Read the bytecode.rs file to get instruction names
with open('crates/compiler-core/src/bytecode.rs', 'r') as f:
    content = f.read()

# Find the Instruction enum
match = re.search(r'pub enum Instruction \{(.+?)\n\}', content, re.DOTALL)
if not match:
    raise ValueError("Could not find Instruction enum")

enum_body = match.group(1)

# Extract variant names
variants = []
for line in enum_body.split('\n'):
    if line.strip().startswith('///') or line.strip().startswith('//'):
        continue
    m = re.match(r'^\s+([A-Z][a-zA-Z0-9]*)', line)
    if m:
        variants.append(m.group(1))

print(f"Found {len(variants)} instruction variants")

# Map RustPython variant names to CPython-compatible names
# The opcode number is the index in the Instruction enum
name_mapping = {
    'BeforeAsyncWith': 'BEFORE_ASYNC_WITH',
    'BeforeWith': 'BEFORE_WITH',
    'BinaryOp': 'BINARY_OP',
    'BinarySubscript': 'BINARY_SUBSCR',
    'Break': 'BREAK',
    'BuildListFromTuples': 'BUILD_LIST_FROM_TUPLES',
    'BuildList': 'BUILD_LIST',
    'BuildMapForCall': 'BUILD_MAP_FOR_CALL',
    'BuildMap': 'BUILD_MAP',
    'BuildSetFromTuples': 'BUILD_SET_FROM_TUPLES',
    'BuildSet': 'BUILD_SET',
    'BuildSlice': 'BUILD_SLICE',
    'BuildString': 'BUILD_STRING',
    'BuildTupleFromIter': 'BUILD_TUPLE_FROM_ITER',
    'BuildTupleFromTuples': 'BUILD_TUPLE_FROM_TUPLES',
    'BuildTuple': 'BUILD_TUPLE',
    'CallFunctionEx': 'CALL_FUNCTION_EX',
    'CallFunctionKeyword': 'CALL_KW',
    'CallFunctionPositional': 'CALL',
    'CallIntrinsic1': 'CALL_INTRINSIC_1',
    'CallIntrinsic2': 'CALL_INTRINSIC_2',
    'CallMethodEx': 'CALL_METHOD_EX',
    'CallMethodKeyword': 'CALL_METHOD_KW',
    'CallMethodPositional': 'CALL_METHOD',
    'CheckEgMatch': 'CHECK_EG_MATCH',
    'CompareOperation': 'COMPARE_OP',
    'ContainsOp': 'CONTAINS_OP',
    'Continue': 'CONTINUE',
    'ConvertValue': 'CONVERT_VALUE',
    'CopyItem': 'COPY',
    'DeleteAttr': 'DELETE_ATTR',
    'DeleteDeref': 'DELETE_DEREF',
    'DeleteFast': 'DELETE_FAST',
    'DeleteGlobal': 'DELETE_GLOBAL',
    'DeleteLocal': 'DELETE_NAME',
    'DeleteSubscript': 'DELETE_SUBSCR',
    'DictUpdate': 'DICT_UPDATE',
    'EndAsyncFor': 'END_ASYNC_FOR',
    'EndFinally': 'END_FINALLY',
    'EnterFinally': 'ENTER_FINALLY',
    'ExtendedArg': 'EXTENDED_ARG',
    'ForIter': 'FOR_ITER',
    'FormatSimple': 'FORMAT_SIMPLE',
    'FormatWithSpec': 'FORMAT_WITH_SPEC',
    'GetAIter': 'GET_AITER',
    'GetANext': 'GET_ANEXT',
    'GetAwaitable': 'GET_AWAITABLE',
    'GetIter': 'GET_ITER',
    'GetLen': 'GET_LEN',
    'ImportFrom': 'IMPORT_FROM',
    'ImportName': 'IMPORT_NAME',
    'IsOp': 'IS_OP',
    'JumpIfFalseOrPop': 'JUMP_IF_FALSE_OR_POP',
    'JumpIfNotExcMatch': 'JUMP_IF_NOT_EXC_MATCH',
    'JumpIfTrueOrPop': 'JUMP_IF_TRUE_OR_POP',
    'Jump': 'JUMP',
    'ListAppend': 'LIST_APPEND',
    'LoadAttr': 'LOAD_ATTR',
    'LoadBuildClass': 'LOAD_BUILD_CLASS',
    'LoadClassDeref': 'LOAD_CLASSDEREF',
    'LoadClosure': 'LOAD_CLOSURE',
    'LoadConst': 'LOAD_CONST',
    'LoadDeref': 'LOAD_DEREF',
    'LoadFast': 'LOAD_FAST',
    'LoadFastAndClear': 'LOAD_FAST_AND_CLEAR',
    'LoadGlobal': 'LOAD_GLOBAL',
    'LoadMethod': 'LOAD_METHOD',
    'LoadNameAny': 'LOAD_NAME',
    'MakeFunction': 'MAKE_FUNCTION',
    'MapAdd': 'MAP_ADD',
    'MatchClass': 'MATCH_CLASS',
    'MatchKeys': 'MATCH_KEYS',
    'MatchMapping': 'MATCH_MAPPING',
    'MatchSequence': 'MATCH_SEQUENCE',
    'Nop': 'NOP',
    'PopBlock': 'POP_BLOCK',
    'PopException': 'POP_EXCEPT',
    'PopJumpIfFalse': 'POP_JUMP_IF_FALSE',
    'PopJumpIfTrue': 'POP_JUMP_IF_TRUE',
    'PopTop': 'POP_TOP',
    'Raise': 'RAISE_VARARGS',
    'Resume': 'RESUME',
    'ReturnConst': 'RETURN_CONST',
    'ReturnValue': 'RETURN_VALUE',
    'Reverse': 'REVERSE',
    'SetAdd': 'SET_ADD',
    'SetFunctionAttribute': 'SET_FUNCTION_ATTRIBUTE',
    'SetupAnnotation': 'SETUP_ANNOTATIONS',
    'SetupExcept': 'SETUP_EXCEPT',
    'SetupFinally': 'SETUP_FINALLY',
    'SetupLoop': 'SETUP_LOOP',
    'StoreAttr': 'STORE_ATTR',
    'StoreDeref': 'STORE_DEREF',
    'StoreFast': 'STORE_FAST',
    'StoreFastLoadFast': 'STORE_FAST_LOAD_FAST',
    'StoreGlobal': 'STORE_GLOBAL',
    'StoreLocal': 'STORE_NAME',
    'StoreSubscript': 'STORE_SUBSCR',
    'Subscript': 'SUBSCRIPT',
    'Swap': 'SWAP',
    'ToBool': 'TO_BOOL',
    'UnaryOperation': 'UNARY_OP',
    'UnpackEx': 'UNPACK_EX',
    'UnpackSequence': 'UNPACK_SEQUENCE',
    'WithExceptStart': 'WITH_EXCEPT_START',
    'YieldFrom': 'YIELD_FROM',
    'YieldValue': 'YIELD_VALUE',
    'Send': 'SEND',
    'EndSend': 'END_SEND',
    'CleanupThrow': 'CLEANUP_THROW',
    'SetExcInfo': 'SET_EXC_INFO',
    'PushExcInfo': 'PUSH_EXC_INFO',
    'CheckExcMatch': 'CHECK_EXC_MATCH',
    'Reraise': 'RERAISE',
}

# Build opmap with RustPython instruction indices
opmap = {}
rust_to_cpython_name = {}
for i, variant in enumerate(variants):
    cpython_name = name_mapping.get(variant, variant.upper())
    opmap[cpython_name] = i
    rust_to_cpython_name[variant] = cpython_name

# Find specific instruction indices for categorization
def find_opcode(cpython_name):
    return opmap.get(cpython_name, -1)

# Generate the output file
output = '''# This file is generated by generate_opcode_metadata.py
# for RustPython bytecode format.
# Do not edit!

# RustPython uses its own bytecode format with sequential opcode numbers.
# This file maps RustPython instruction names to their indices in the Instruction enum.

_specializations = {}

_specialized_opmap = {}

opmap = {
'''

for name, num in sorted(opmap.items(), key=lambda x: x[1]):
    output += f"    '{name}': {num},\n"

# Add pseudo-ops for CPython compatibility (these don't exist in RustPython)
# but are needed for dis.py to not crash
pseudo_ops = {
    'CACHE': 256,
    'ENTER_EXECUTOR': 257,
    'JUMP_BACKWARD': 258,
    'JUMP_BACKWARD_NO_INTERRUPT': 259,
    'LOAD_SUPER_ATTR': 260,
    'LOAD_FAST_LOAD_FAST': 261,
    'STORE_FAST_STORE_FAST': 262,
    'JUMP_FORWARD': 263,
    'END_FOR': 264,
    'INTERPRETER_EXIT': 265,
    'LOAD_ASSERTION_ERROR': 266,
    'LOAD_LOCALS': 267,
    'BINARY_SLICE': 268,
    'STORE_SLICE': 269,
    'PUSH_NULL': 270,
    'RETURN_GENERATOR': 271,
    'GET_YIELD_FROM_ITER': 272,
    'COPY_FREE_VARS': 273,
    'LOAD_FAST_CHECK': 274,
    'LOAD_FROM_DICT_OR_DEREF': 275,
    'LOAD_FROM_DICT_OR_GLOBALS': 276,
    'DICT_MERGE': 277,
    'LIST_EXTEND': 278,
    'SET_UPDATE': 279,
    'MAKE_CELL': 280,
    'EXIT_INIT_CHECK': 281,
    'POP_JUMP_IF_NONE': 282,
    'POP_JUMP_IF_NOT_NONE': 283,
    'STORE_FAST_MAYBE_NULL': 284,
    'SETUP_CLEANUP': 285,
    'SETUP_WITH': 286,
    'JUMP_NO_INTERRUPT': 287,
    'UNARY_NEGATIVE': 288,
    'UNARY_INVERT': 289,
    'UNARY_NOT': 290,
    'BUILD_CONST_KEY_MAP': 291,
    'RESERVED': 292,
}

for name, num in sorted(pseudo_ops.items(), key=lambda x: x[1]):
    if name not in opmap:  # Only add if not already defined
        output += f"    '{name}': {num},\n"

output += '''}

# HAVE_ARGUMENT is not used in RustPython (all ops can have args)
HAVE_ARGUMENT = 0
MIN_INSTRUMENTED_OPCODE = 256
'''

with open('Lib/_opcode_metadata.py', 'w') as f:
    f.write(output)

print("Generated Lib/_opcode_metadata.py")
print(f"\nKey opcode indices (for opcode.rs):")
print(f"  LOAD_CONST = {find_opcode('LOAD_CONST')}")
print(f"  RETURN_CONST = {find_opcode('RETURN_CONST')}")
print(f"  LOAD_GLOBAL = {find_opcode('LOAD_GLOBAL')}")
print(f"  LOAD_NAME = {find_opcode('LOAD_NAME')}")
print(f"  LOAD_ATTR = {find_opcode('LOAD_ATTR')}")
print(f"  STORE_NAME = {find_opcode('STORE_NAME')}")
print(f"  STORE_FAST = {find_opcode('STORE_FAST')}")
print(f"  LOAD_FAST = {find_opcode('LOAD_FAST')}")
print(f"  FOR_ITER = {find_opcode('FOR_ITER')}")
print(f"  POP_JUMP_IF_FALSE = {find_opcode('POP_JUMP_IF_FALSE')}")
