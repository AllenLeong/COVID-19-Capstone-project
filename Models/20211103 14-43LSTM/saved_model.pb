ю&
Ы
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ћ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleщelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleщelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ыЮ%
z
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

:  *
dtype0
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
: *
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

: *
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_71/lstm_cell_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_71/lstm_cell_71/kernel

/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/kernel*
_output_shapes
:	*
dtype0
Ї
%lstm_71/lstm_cell_71/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%lstm_71/lstm_cell_71/recurrent_kernel
 
9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_71/lstm_cell_71/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_71/lstm_cell_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_71/lstm_cell_71/bias

-lstm_71/lstm_cell_71/bias/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_86/kernel/m

*Adam/dense_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_86/bias/m
y
(Adam/dense_86/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/m*
_output_shapes
: *
dtype0

Adam/dense_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_87/kernel/m

*Adam/dense_87/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/m
y
(Adam/dense_87/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/m*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_71/lstm_cell_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/m

6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/m*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
Ў
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

 Adam/lstm_71/lstm_cell_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/m

4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_86/kernel/v

*Adam/dense_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_86/bias/v
y
(Adam/dense_86/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/v*
_output_shapes
: *
dtype0

Adam/dense_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_87/kernel/v

*Adam/dense_87/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/v
y
(Adam/dense_87/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/v*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_71/lstm_cell_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/v

6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/v*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
Ў
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

 Adam/lstm_71/lstm_cell_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/v

4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
З,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ђ+
valueш+Bх+ Bо+
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
 	keras_api
О
!iter

"beta_1

#beta_2
	$decay
%learning_ratemRmSmTmU&mV'mW(mXvYvZv[v\&v]'v^(v_
1
&0
'1
(2
3
4
5
6
1
&0
'1
(2
3
4
5
6
 
­

)layers
trainable_variables
	variables
*metrics
+layer_metrics
,layer_regularization_losses
-non_trainable_variables
regularization_losses
 

.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
 

&0
'1
(2

&0
'1
(2
 
Й

3layers
trainable_variables
	variables
4metrics
5layer_metrics
6layer_regularization_losses
7non_trainable_variables

8states
regularization_losses
[Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_86/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

9layers
trainable_variables
	variables
:metrics
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_87/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

>layers
trainable_variables
	variables
?metrics
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses
 
 
 
­

Clayers
trainable_variables
	variables
Dmetrics
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_71/lstm_cell_71/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_71/lstm_cell_71/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_71/lstm_cell_71/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

H0
 
 
 
 

&0
'1
(2

&0
'1
(2
 
­

Ilayers
/trainable_variables
0	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
Mnon_trainable_variables
1regularization_losses

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ntotal
	Ocount
P	variables
Q	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

P	variables
~|
VARIABLE_VALUEAdam/dense_86/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_86/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_87/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_87/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_86/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_86/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_87/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_87/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_30Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_30lstm_71/lstm_cell_71/kernellstm_71/lstm_cell_71/bias%lstm_71/lstm_cell_71/recurrent_kerneldense_86/kerneldense_86/biasdense_87/kerneldense_87/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_2304209
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
њ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOp9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOp-lstm_71/lstm_cell_71/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_86/kernel/m/Read/ReadVariableOp(Adam/dense_86/bias/m/Read/ReadVariableOp*Adam/dense_87/kernel/m/Read/ReadVariableOp(Adam/dense_87/bias/m/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOp*Adam/dense_86/kernel/v/Read/ReadVariableOp(Adam/dense_86/bias/v/Read/ReadVariableOp*Adam/dense_87/kernel/v/Read/ReadVariableOp(Adam/dense_87/bias/v/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_2306435
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_86/kerneldense_86/biasdense_87/kerneldense_87/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_71/lstm_cell_71/kernel%lstm_71/lstm_cell_71/recurrent_kernellstm_71/lstm_cell_71/biastotalcountAdam/dense_86/kernel/mAdam/dense_86/bias/mAdam/dense_87/kernel/mAdam/dense_87/bias/m"Adam/lstm_71/lstm_cell_71/kernel/m,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m Adam/lstm_71/lstm_cell_71/bias/mAdam/dense_86/kernel/vAdam/dense_86/bias/vAdam/dense_87/kernel/vAdam/dense_87/bias/v"Adam/lstm_71/lstm_cell_71/kernel/v,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v Adam/lstm_71/lstm_cell_71/bias/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_2306529шЯ$

c
G__inference_reshape_43_layer_call_and_return_conditional_losses_2306072

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
R
Щ
D__inference_lstm_71_layer_call_and_return_conditional_losses_2303041

inputs'
lstm_cell_71_2302953:	#
lstm_cell_71_2302955:	'
lstm_cell_71_2302957:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_71/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ё
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_2302953lstm_cell_71_2302955lstm_cell_71_2302957*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23028882&
$lstm_cell_71/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_2302953lstm_cell_71_2302955lstm_cell_71_2302957*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2302966*
condR
while_cond_2302965*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeд
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_71_2302953*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_71/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
Њ
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305453
inputs_0=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like}
lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout/ConstГ
lstm_cell_71/dropout/MulMullstm_cell_71/ones_like:output:0#lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul
lstm_cell_71/dropout/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout/Shapeј
1lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЎЛ23
1lstm_cell_71/dropout/random_uniform/RandomUniform
#lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_71/dropout/GreaterEqual/yђ
!lstm_cell_71/dropout/GreaterEqualGreaterEqual:lstm_cell_71/dropout/random_uniform/RandomUniform:output:0,lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_71/dropout/GreaterEqualІ
lstm_cell_71/dropout/CastCast%lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/CastЎ
lstm_cell_71/dropout/Mul_1Mullstm_cell_71/dropout/Mul:z:0lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul_1
lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_1/ConstЙ
lstm_cell_71/dropout_1/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul
lstm_cell_71/dropout_1/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_1/Shape§
3lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2їЬd25
3lstm_cell_71/dropout_1/random_uniform/RandomUniform
%lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_1/GreaterEqual/yњ
#lstm_cell_71/dropout_1/GreaterEqualGreaterEqual<lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_1/GreaterEqualЌ
lstm_cell_71/dropout_1/CastCast'lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/CastЖ
lstm_cell_71/dropout_1/Mul_1Mullstm_cell_71/dropout_1/Mul:z:0lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul_1
lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_2/ConstЙ
lstm_cell_71/dropout_2/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul
lstm_cell_71/dropout_2/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_2/Shapeў
3lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Тъ25
3lstm_cell_71/dropout_2/random_uniform/RandomUniform
%lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_2/GreaterEqual/yњ
#lstm_cell_71/dropout_2/GreaterEqualGreaterEqual<lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_2/GreaterEqualЌ
lstm_cell_71/dropout_2/CastCast'lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/CastЖ
lstm_cell_71/dropout_2/Mul_1Mullstm_cell_71/dropout_2/Mul:z:0lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul_1
lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_3/ConstЙ
lstm_cell_71/dropout_3/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul
lstm_cell_71/dropout_3/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_3/Shapeў
3lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Дщи25
3lstm_cell_71/dropout_3/random_uniform/RandomUniform
%lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_3/GreaterEqual/yњ
#lstm_cell_71/dropout_3/GreaterEqualGreaterEqual<lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_3/GreaterEqualЌ
lstm_cell_71/dropout_3/CastCast'lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/CastЖ
lstm_cell_71/dropout_3/Mul_1Mullstm_cell_71/dropout_3/Mul:z:0lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul_1~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0 lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0 lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0 lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2305288*
condR
while_cond_2305287*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
кЯ
Ј
D__inference_lstm_71_layer_call_and_return_conditional_losses_2306003

inputs=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like}
lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout/ConstГ
lstm_cell_71/dropout/MulMullstm_cell_71/ones_like:output:0#lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul
lstm_cell_71/dropout/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout/Shapeј
1lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2т 23
1lstm_cell_71/dropout/random_uniform/RandomUniform
#lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_71/dropout/GreaterEqual/yђ
!lstm_cell_71/dropout/GreaterEqualGreaterEqual:lstm_cell_71/dropout/random_uniform/RandomUniform:output:0,lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_71/dropout/GreaterEqualІ
lstm_cell_71/dropout/CastCast%lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/CastЎ
lstm_cell_71/dropout/Mul_1Mullstm_cell_71/dropout/Mul:z:0lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul_1
lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_1/ConstЙ
lstm_cell_71/dropout_1/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul
lstm_cell_71/dropout_1/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_1/Shapeў
3lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ђ25
3lstm_cell_71/dropout_1/random_uniform/RandomUniform
%lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_1/GreaterEqual/yњ
#lstm_cell_71/dropout_1/GreaterEqualGreaterEqual<lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_1/GreaterEqualЌ
lstm_cell_71/dropout_1/CastCast'lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/CastЖ
lstm_cell_71/dropout_1/Mul_1Mullstm_cell_71/dropout_1/Mul:z:0lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul_1
lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_2/ConstЙ
lstm_cell_71/dropout_2/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul
lstm_cell_71/dropout_2/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_2/Shape§
3lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ќі(25
3lstm_cell_71/dropout_2/random_uniform/RandomUniform
%lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_2/GreaterEqual/yњ
#lstm_cell_71/dropout_2/GreaterEqualGreaterEqual<lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_2/GreaterEqualЌ
lstm_cell_71/dropout_2/CastCast'lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/CastЖ
lstm_cell_71/dropout_2/Mul_1Mullstm_cell_71/dropout_2/Mul:z:0lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul_1
lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_3/ConstЙ
lstm_cell_71/dropout_3/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul
lstm_cell_71/dropout_3/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_3/Shape§
3lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2дЗ
25
3lstm_cell_71/dropout_3/random_uniform/RandomUniform
%lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_3/GreaterEqual/yњ
#lstm_cell_71/dropout_3/GreaterEqualGreaterEqual<lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_3/GreaterEqualЌ
lstm_cell_71/dropout_3/CastCast'lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/CastЖ
lstm_cell_71/dropout_3/Mul_1Mullstm_cell_71/dropout_3/Mul:z:0lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul_1~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0 lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0 lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0 lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2305838*
condR
while_cond_2305837*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с

J__inference_sequential_29_layer_call_and_return_conditional_losses_2304518

inputsE
2lstm_71_lstm_cell_71_split_readvariableop_resource:	C
4lstm_71_lstm_cell_71_split_1_readvariableop_resource:	?
,lstm_71_lstm_cell_71_readvariableop_resource:	 9
'dense_86_matmul_readvariableop_resource:  6
(dense_86_biasadd_readvariableop_resource: 9
'dense_87_matmul_readvariableop_resource: 6
(dense_87_biasadd_readvariableop_resource:
identityЂdense_86/BiasAdd/ReadVariableOpЂdense_86/MatMul/ReadVariableOpЂdense_87/BiasAdd/ReadVariableOpЂdense_87/MatMul/ReadVariableOpЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂ#lstm_71/lstm_cell_71/ReadVariableOpЂ%lstm_71/lstm_cell_71/ReadVariableOp_1Ђ%lstm_71/lstm_cell_71/ReadVariableOp_2Ђ%lstm_71/lstm_cell_71/ReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_71/lstm_cell_71/split/ReadVariableOpЂ+lstm_71/lstm_cell_71/split_1/ReadVariableOpЂlstm_71/whileT
lstm_71/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_71/Shape
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice/stack
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_1
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_2
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slicel
lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros/mul/y
lstm_71/zeros/mulMullstm_71/strided_slice:output:0lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/mulo
lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_71/zeros/Less/y
lstm_71/zeros/LessLesslstm_71/zeros/mul:z:0lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/Lessr
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros/packed/1Ѓ
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros/packedo
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros/Const
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/zerosp
lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros_1/mul/y
lstm_71/zeros_1/mulMullstm_71/strided_slice:output:0lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/muls
lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_71/zeros_1/Less/y
lstm_71/zeros_1/LessLesslstm_71/zeros_1/mul:z:0lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/Lessv
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros_1/packed/1Љ
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros_1/packeds
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros_1/Const
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/zeros_1
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose/perm
lstm_71/transpose	Transposeinputslstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_71/transposeg
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:2
lstm_71/Shape_1
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_1/stack
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_1
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_2
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slice_1
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_71/TensorArrayV2/element_shapeв
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2Я
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_71/TensorArrayUnstack/TensorListFromTensor
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_2/stack
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_1
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_2Ќ
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_71/strided_slice_2
$lstm_71/lstm_cell_71/ones_like/ShapeShapelstm_71/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_71/lstm_cell_71/ones_like/Shape
$lstm_71/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_71/lstm_cell_71/ones_like/Constи
lstm_71/lstm_cell_71/ones_likeFill-lstm_71/lstm_cell_71/ones_like/Shape:output:0-lstm_71/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/ones_like
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_71/lstm_cell_71/split/split_dimЪ
)lstm_71/lstm_cell_71/split/ReadVariableOpReadVariableOp2lstm_71_lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_71/lstm_cell_71/split/ReadVariableOpћ
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:01lstm_71/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_71/lstm_cell_71/splitН
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMulС
lstm_71/lstm_cell_71/MatMul_1MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_1С
lstm_71/lstm_cell_71/MatMul_2MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_2С
lstm_71/lstm_cell_71/MatMul_3MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_3
&lstm_71/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_71/lstm_cell_71/split_1/split_dimЬ
+lstm_71/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_71/lstm_cell_71/split_1/ReadVariableOpѓ
lstm_71/lstm_cell_71/split_1Split/lstm_71/lstm_cell_71/split_1/split_dim:output:03lstm_71/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_71/lstm_cell_71/split_1Ч
lstm_71/lstm_cell_71/BiasAddBiasAdd%lstm_71/lstm_cell_71/MatMul:product:0%lstm_71/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/BiasAddЭ
lstm_71/lstm_cell_71/BiasAdd_1BiasAdd'lstm_71/lstm_cell_71/MatMul_1:product:0%lstm_71/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_1Э
lstm_71/lstm_cell_71/BiasAdd_2BiasAdd'lstm_71/lstm_cell_71/MatMul_2:product:0%lstm_71/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_2Э
lstm_71/lstm_cell_71/BiasAdd_3BiasAdd'lstm_71/lstm_cell_71/MatMul_3:product:0%lstm_71/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_3Ў
lstm_71/lstm_cell_71/mulMullstm_71/zeros:output:0'lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mulВ
lstm_71/lstm_cell_71/mul_1Mullstm_71/zeros:output:0'lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_1В
lstm_71/lstm_cell_71/mul_2Mullstm_71/zeros:output:0'lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_2В
lstm_71/lstm_cell_71/mul_3Mullstm_71/zeros:output:0'lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_3И
#lstm_71/lstm_cell_71/ReadVariableOpReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_71/lstm_cell_71/ReadVariableOpЅ
(lstm_71/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_71/lstm_cell_71/strided_slice/stackЉ
*lstm_71/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_71/lstm_cell_71/strided_slice/stack_1Љ
*lstm_71/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_71/lstm_cell_71/strided_slice/stack_2њ
"lstm_71/lstm_cell_71/strided_sliceStridedSlice+lstm_71/lstm_cell_71/ReadVariableOp:value:01lstm_71/lstm_cell_71/strided_slice/stack:output:03lstm_71/lstm_cell_71/strided_slice/stack_1:output:03lstm_71/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_71/lstm_cell_71/strided_sliceХ
lstm_71/lstm_cell_71/MatMul_4MatMullstm_71/lstm_cell_71/mul:z:0+lstm_71/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_4П
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/BiasAdd:output:0'lstm_71/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add
lstm_71/lstm_cell_71/SigmoidSigmoidlstm_71/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/SigmoidМ
%lstm_71/lstm_cell_71/ReadVariableOp_1ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_1Љ
*lstm_71/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_71/lstm_cell_71/strided_slice_1/stack­
,lstm_71/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_71/lstm_cell_71/strided_slice_1/stack_1­
,lstm_71/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_1/stack_2
$lstm_71/lstm_cell_71/strided_slice_1StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_1:value:03lstm_71/lstm_cell_71/strided_slice_1/stack:output:05lstm_71/lstm_cell_71/strided_slice_1/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_1Щ
lstm_71/lstm_cell_71/MatMul_5MatMullstm_71/lstm_cell_71/mul_1:z:0-lstm_71/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_5Х
lstm_71/lstm_cell_71/add_1AddV2'lstm_71/lstm_cell_71/BiasAdd_1:output:0'lstm_71/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_1
lstm_71/lstm_cell_71/Sigmoid_1Sigmoidlstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/Sigmoid_1Џ
lstm_71/lstm_cell_71/mul_4Mul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_4М
%lstm_71/lstm_cell_71/ReadVariableOp_2ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_2Љ
*lstm_71/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_71/lstm_cell_71/strided_slice_2/stack­
,lstm_71/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_71/lstm_cell_71/strided_slice_2/stack_1­
,lstm_71/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_2/stack_2
$lstm_71/lstm_cell_71/strided_slice_2StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_2:value:03lstm_71/lstm_cell_71/strided_slice_2/stack:output:05lstm_71/lstm_cell_71/strided_slice_2/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_2Щ
lstm_71/lstm_cell_71/MatMul_6MatMullstm_71/lstm_cell_71/mul_2:z:0-lstm_71/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_6Х
lstm_71/lstm_cell_71/add_2AddV2'lstm_71/lstm_cell_71/BiasAdd_2:output:0'lstm_71/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_2
lstm_71/lstm_cell_71/ReluRelulstm_71/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/ReluМ
lstm_71/lstm_cell_71/mul_5Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_5Г
lstm_71/lstm_cell_71/add_3AddV2lstm_71/lstm_cell_71/mul_4:z:0lstm_71/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_3М
%lstm_71/lstm_cell_71/ReadVariableOp_3ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_3Љ
*lstm_71/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_71/lstm_cell_71/strided_slice_3/stack­
,lstm_71/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_71/lstm_cell_71/strided_slice_3/stack_1­
,lstm_71/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_3/stack_2
$lstm_71/lstm_cell_71/strided_slice_3StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_3:value:03lstm_71/lstm_cell_71/strided_slice_3/stack:output:05lstm_71/lstm_cell_71/strided_slice_3/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_3Щ
lstm_71/lstm_cell_71/MatMul_7MatMullstm_71/lstm_cell_71/mul_3:z:0-lstm_71/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_7Х
lstm_71/lstm_cell_71/add_4AddV2'lstm_71/lstm_cell_71/BiasAdd_3:output:0'lstm_71/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_4
lstm_71/lstm_cell_71/Sigmoid_2Sigmoidlstm_71/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/Sigmoid_2
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/Relu_1Р
lstm_71/lstm_cell_71/mul_6Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_6
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_71/TensorArrayV2_1/element_shapeи
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2_1^
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/time
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_71/while/maximum_iterationsz
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/while/loop_counterћ
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_71_lstm_cell_71_split_readvariableop_resource4lstm_71_lstm_cell_71_split_1_readvariableop_resource,lstm_71_lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_71_while_body_2304357*&
condR
lstm_71_while_cond_2304356*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_71/whileХ
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_71/TensorArrayV2Stack/TensorListStack
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_71/strided_slice_3/stack
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_71/strided_slice_3/stack_1
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_3/stack_2Ъ
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_71/strided_slice_3
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose_1/permХ
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_71/transpose_1v
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/runtimeЈ
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_86/MatMul/ReadVariableOpЈ
dense_86/MatMulMatMul lstm_71/strided_slice_3:output:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/MatMulЇ
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_86/BiasAdd/ReadVariableOpЅ
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/ReluЈ
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_87/MatMul/ReadVariableOpЃ
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_87/MatMulЇ
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOpЅ
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_87/BiasAddm
reshape_43/ShapeShapedense_87/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_43/Shape
reshape_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_43/strided_slice/stack
 reshape_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_43/strided_slice/stack_1
 reshape_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_43/strided_slice/stack_2Є
reshape_43/strided_sliceStridedSlicereshape_43/Shape:output:0'reshape_43/strided_slice/stack:output:0)reshape_43/strided_slice/stack_1:output:0)reshape_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_43/strided_slicez
reshape_43/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_43/Reshape/shape/1z
reshape_43/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_43/Reshape/shape/2з
reshape_43/Reshape/shapePack!reshape_43/strided_slice:output:0#reshape_43/Reshape/shape/1:output:0#reshape_43/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_43/Reshape/shapeЇ
reshape_43/ReshapeReshapedense_87/BiasAdd:output:0!reshape_43/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_43/Reshapeђ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_71_lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЧ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mulz
IdentityIdentityreshape_43/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp0^dense_87/bias/Regularizer/Square/ReadVariableOp$^lstm_71/lstm_cell_71/ReadVariableOp&^lstm_71/lstm_cell_71/ReadVariableOp_1&^lstm_71/lstm_cell_71/ReadVariableOp_2&^lstm_71/lstm_cell_71/ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*^lstm_71/lstm_cell_71/split/ReadVariableOp,^lstm_71/lstm_cell_71/split_1/ReadVariableOp^lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2J
#lstm_71/lstm_cell_71/ReadVariableOp#lstm_71/lstm_cell_71/ReadVariableOp2N
%lstm_71/lstm_cell_71/ReadVariableOp_1%lstm_71/lstm_cell_71/ReadVariableOp_12N
%lstm_71/lstm_cell_71/ReadVariableOp_2%lstm_71/lstm_cell_71/ReadVariableOp_22N
%lstm_71/lstm_cell_71/ReadVariableOp_3%lstm_71/lstm_cell_71/ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_71/lstm_cell_71/split/ReadVariableOp)lstm_71/lstm_cell_71/split/ReadVariableOp2Z
+lstm_71/lstm_cell_71/split_1/ReadVariableOp+lstm_71/lstm_cell_71/split_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Иv
ь
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306317

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeб
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2љтй2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeз
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2 Ћ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeз
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2сЦ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeз
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ъГ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6н
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
Й
Љ
(sequential_29_lstm_71_while_body_2302382H
Dsequential_29_lstm_71_while_sequential_29_lstm_71_while_loop_counterN
Jsequential_29_lstm_71_while_sequential_29_lstm_71_while_maximum_iterations+
'sequential_29_lstm_71_while_placeholder-
)sequential_29_lstm_71_while_placeholder_1-
)sequential_29_lstm_71_while_placeholder_2-
)sequential_29_lstm_71_while_placeholder_3G
Csequential_29_lstm_71_while_sequential_29_lstm_71_strided_slice_1_0
sequential_29_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_29_lstm_71_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_29_lstm_71_while_lstm_cell_71_split_readvariableop_resource_0:	Y
Jsequential_29_lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0:	U
Bsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0:	 (
$sequential_29_lstm_71_while_identity*
&sequential_29_lstm_71_while_identity_1*
&sequential_29_lstm_71_while_identity_2*
&sequential_29_lstm_71_while_identity_3*
&sequential_29_lstm_71_while_identity_4*
&sequential_29_lstm_71_while_identity_5E
Asequential_29_lstm_71_while_sequential_29_lstm_71_strided_slice_1
}sequential_29_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_29_lstm_71_tensorarrayunstack_tensorlistfromtensorY
Fsequential_29_lstm_71_while_lstm_cell_71_split_readvariableop_resource:	W
Hsequential_29_lstm_71_while_lstm_cell_71_split_1_readvariableop_resource:	S
@sequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource:	 Ђ7sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOpЂ9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_1Ђ9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_2Ђ9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_3Ђ=sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOpЂ?sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOpя
Msequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2O
Msequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeз
?sequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_29_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_29_lstm_71_tensorarrayunstack_tensorlistfromtensor_0'sequential_29_lstm_71_while_placeholderVsequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02A
?sequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItemЭ
8sequential_29/lstm_71/while/lstm_cell_71/ones_like/ShapeShape)sequential_29_lstm_71_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_29/lstm_71/while/lstm_cell_71/ones_like/ShapeЙ
8sequential_29/lstm_71/while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_29/lstm_71/while/lstm_cell_71/ones_like/ConstЈ
2sequential_29/lstm_71/while/lstm_cell_71/ones_likeFillAsequential_29/lstm_71/while/lstm_cell_71/ones_like/Shape:output:0Asequential_29/lstm_71/while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/ones_likeЖ
8sequential_29/lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_29/lstm_71/while/lstm_cell_71/split/split_dim
=sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOpReadVariableOpHsequential_29_lstm_71_while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02?
=sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOpЫ
.sequential_29/lstm_71/while/lstm_cell_71/splitSplitAsequential_29/lstm_71/while/lstm_cell_71/split/split_dim:output:0Esequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split20
.sequential_29/lstm_71/while/lstm_cell_71/split
/sequential_29/lstm_71/while/lstm_cell_71/MatMulMatMulFsequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_29/lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_29/lstm_71/while/lstm_cell_71/MatMulЃ
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_1MatMulFsequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_29/lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_1Ѓ
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_2MatMulFsequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_29/lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_2Ѓ
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_3MatMulFsequential_29/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_29/lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_3К
:sequential_29/lstm_71/while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_29/lstm_71/while/lstm_cell_71/split_1/split_dim
?sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOpReadVariableOpJsequential_29_lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOpУ
0sequential_29/lstm_71/while/lstm_cell_71/split_1SplitCsequential_29/lstm_71/while/lstm_cell_71/split_1/split_dim:output:0Gsequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split22
0sequential_29/lstm_71/while/lstm_cell_71/split_1
0sequential_29/lstm_71/while/lstm_cell_71/BiasAddBiasAdd9sequential_29/lstm_71/while/lstm_cell_71/MatMul:product:09sequential_29/lstm_71/while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_29/lstm_71/while/lstm_cell_71/BiasAdd
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_1BiasAdd;sequential_29/lstm_71/while/lstm_cell_71/MatMul_1:product:09sequential_29/lstm_71/while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_1
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_2BiasAdd;sequential_29/lstm_71/while/lstm_cell_71/MatMul_2:product:09sequential_29/lstm_71/while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_2
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_3BiasAdd;sequential_29/lstm_71/while/lstm_cell_71/MatMul_3:product:09sequential_29/lstm_71/while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_3§
,sequential_29/lstm_71/while/lstm_cell_71/mulMul)sequential_29_lstm_71_while_placeholder_2;sequential_29/lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/while/lstm_cell_71/mul
.sequential_29/lstm_71/while/lstm_cell_71/mul_1Mul)sequential_29_lstm_71_while_placeholder_2;sequential_29/lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_1
.sequential_29/lstm_71/while/lstm_cell_71/mul_2Mul)sequential_29_lstm_71_while_placeholder_2;sequential_29/lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_2
.sequential_29/lstm_71/while/lstm_cell_71/mul_3Mul)sequential_29_lstm_71_while_placeholder_2;sequential_29/lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_3і
7sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOpReadVariableOpBsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype029
7sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOpЭ
<sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stackб
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_1б
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_2ђ
6sequential_29/lstm_71/while/lstm_cell_71/strided_sliceStridedSlice?sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp:value:0Esequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack:output:0Gsequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_1:output:0Gsequential_29/lstm_71/while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_29/lstm_71/while/lstm_cell_71/strided_slice
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_4MatMul0sequential_29/lstm_71/while/lstm_cell_71/mul:z:0?sequential_29/lstm_71/while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_4
,sequential_29/lstm_71/while/lstm_cell_71/addAddV29sequential_29/lstm_71/while/lstm_cell_71/BiasAdd:output:0;sequential_29/lstm_71/while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/while/lstm_cell_71/addг
0sequential_29/lstm_71/while/lstm_cell_71/SigmoidSigmoid0sequential_29/lstm_71/while/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_29/lstm_71/while/lstm_cell_71/Sigmoidњ
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_1ReadVariableOpBsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_1б
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stackе
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_1е
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_2ў
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1StridedSliceAsequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_1:value:0Gsequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_1:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_1
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_5MatMul2sequential_29/lstm_71/while/lstm_cell_71/mul_1:z:0Asequential_29/lstm_71/while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_5
.sequential_29/lstm_71/while/lstm_cell_71/add_1AddV2;sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_1:output:0;sequential_29/lstm_71/while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/add_1й
2sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid2sequential_29/lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_1ќ
.sequential_29/lstm_71/while/lstm_cell_71/mul_4Mul6sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_1:y:0)sequential_29_lstm_71_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_4њ
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_2ReadVariableOpBsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_2б
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stackе
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_1е
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_2ў
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2StridedSliceAsequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_2:value:0Gsequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_1:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_2
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_6MatMul2sequential_29/lstm_71/while/lstm_cell_71/mul_2:z:0Asequential_29/lstm_71/while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_6
.sequential_29/lstm_71/while/lstm_cell_71/add_2AddV2;sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_2:output:0;sequential_29/lstm_71/while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/add_2Ь
-sequential_29/lstm_71/while/lstm_cell_71/ReluRelu2sequential_29/lstm_71/while/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_29/lstm_71/while/lstm_cell_71/Relu
.sequential_29/lstm_71/while/lstm_cell_71/mul_5Mul4sequential_29/lstm_71/while/lstm_cell_71/Sigmoid:y:0;sequential_29/lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_5
.sequential_29/lstm_71/while/lstm_cell_71/add_3AddV22sequential_29/lstm_71/while/lstm_cell_71/mul_4:z:02sequential_29/lstm_71/while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/add_3њ
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_3ReadVariableOpBsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_3б
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2@
>sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stackе
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_1е
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_2ў
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3StridedSliceAsequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_3:value:0Gsequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_1:output:0Isequential_29/lstm_71/while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_29/lstm_71/while/lstm_cell_71/strided_slice_3
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_7MatMul2sequential_29/lstm_71/while/lstm_cell_71/mul_3:z:0Asequential_29/lstm_71/while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_29/lstm_71/while/lstm_cell_71/MatMul_7
.sequential_29/lstm_71/while/lstm_cell_71/add_4AddV2;sequential_29/lstm_71/while/lstm_cell_71/BiasAdd_3:output:0;sequential_29/lstm_71/while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/add_4й
2sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid2sequential_29/lstm_71/while/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_2а
/sequential_29/lstm_71/while/lstm_cell_71/Relu_1Relu2sequential_29/lstm_71/while/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_29/lstm_71/while/lstm_cell_71/Relu_1
.sequential_29/lstm_71/while/lstm_cell_71/mul_6Mul6sequential_29/lstm_71/while/lstm_cell_71/Sigmoid_2:y:0=sequential_29/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_29/lstm_71/while/lstm_cell_71/mul_6Ю
@sequential_29/lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_29_lstm_71_while_placeholder_1'sequential_29_lstm_71_while_placeholder2sequential_29/lstm_71/while/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02B
@sequential_29/lstm_71/while/TensorArrayV2Write/TensorListSetItem
!sequential_29/lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_29/lstm_71/while/add/yС
sequential_29/lstm_71/while/addAddV2'sequential_29_lstm_71_while_placeholder*sequential_29/lstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_29/lstm_71/while/add
#sequential_29/lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_29/lstm_71/while/add_1/yф
!sequential_29/lstm_71/while/add_1AddV2Dsequential_29_lstm_71_while_sequential_29_lstm_71_while_loop_counter,sequential_29/lstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_29/lstm_71/while/add_1У
$sequential_29/lstm_71/while/IdentityIdentity%sequential_29/lstm_71/while/add_1:z:0!^sequential_29/lstm_71/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_29/lstm_71/while/Identityь
&sequential_29/lstm_71/while/Identity_1IdentityJsequential_29_lstm_71_while_sequential_29_lstm_71_while_maximum_iterations!^sequential_29/lstm_71/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_29/lstm_71/while/Identity_1Х
&sequential_29/lstm_71/while/Identity_2Identity#sequential_29/lstm_71/while/add:z:0!^sequential_29/lstm_71/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_29/lstm_71/while/Identity_2ђ
&sequential_29/lstm_71/while/Identity_3IdentityPsequential_29/lstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_29/lstm_71/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_29/lstm_71/while/Identity_3х
&sequential_29/lstm_71/while/Identity_4Identity2sequential_29/lstm_71/while/lstm_cell_71/mul_6:z:0!^sequential_29/lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_29/lstm_71/while/Identity_4х
&sequential_29/lstm_71/while/Identity_5Identity2sequential_29/lstm_71/while/lstm_cell_71/add_3:z:0!^sequential_29/lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_29/lstm_71/while/Identity_5і
 sequential_29/lstm_71/while/NoOpNoOp8^sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp:^sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_1:^sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_2:^sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_3>^sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOp@^sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_29/lstm_71/while/NoOp"U
$sequential_29_lstm_71_while_identity-sequential_29/lstm_71/while/Identity:output:0"Y
&sequential_29_lstm_71_while_identity_1/sequential_29/lstm_71/while/Identity_1:output:0"Y
&sequential_29_lstm_71_while_identity_2/sequential_29/lstm_71/while/Identity_2:output:0"Y
&sequential_29_lstm_71_while_identity_3/sequential_29/lstm_71/while/Identity_3:output:0"Y
&sequential_29_lstm_71_while_identity_4/sequential_29/lstm_71/while/Identity_4:output:0"Y
&sequential_29_lstm_71_while_identity_5/sequential_29/lstm_71/while/Identity_5:output:0"
@sequential_29_lstm_71_while_lstm_cell_71_readvariableop_resourceBsequential_29_lstm_71_while_lstm_cell_71_readvariableop_resource_0"
Hsequential_29_lstm_71_while_lstm_cell_71_split_1_readvariableop_resourceJsequential_29_lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0"
Fsequential_29_lstm_71_while_lstm_cell_71_split_readvariableop_resourceHsequential_29_lstm_71_while_lstm_cell_71_split_readvariableop_resource_0"
Asequential_29_lstm_71_while_sequential_29_lstm_71_strided_slice_1Csequential_29_lstm_71_while_sequential_29_lstm_71_strided_slice_1_0"
}sequential_29_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_29_lstm_71_tensorarrayunstack_tensorlistfromtensorsequential_29_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_29_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2r
7sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp7sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp2v
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_19sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_12v
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_29sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_22v
9sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_39sequential_29/lstm_71/while/lstm_cell_71/ReadVariableOp_32~
=sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOp=sequential_29/lstm_71/while/lstm_cell_71/split/ReadVariableOp2
?sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOp?sequential_29/lstm_71/while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
к
Ш
while_cond_2305287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2305287___redundant_placeholder05
1while_while_cond_2305287___redundant_placeholder15
1while_while_cond_2305287___redundant_placeholder25
1while_while_cond_2305287___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Ј
Ж
)__inference_lstm_71_layer_call_fn_2304892

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23035642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
Н
lstm_71_while_body_2304357,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0:	K
<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0:	G
4lstm_71_while_lstm_cell_71_readvariableop_resource_0:	 
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorK
8lstm_71_while_lstm_cell_71_split_readvariableop_resource:	I
:lstm_71_while_lstm_cell_71_split_1_readvariableop_resource:	E
2lstm_71_while_lstm_cell_71_readvariableop_resource:	 Ђ)lstm_71/while/lstm_cell_71/ReadVariableOpЂ+lstm_71/while/lstm_cell_71/ReadVariableOp_1Ђ+lstm_71/while/lstm_cell_71/ReadVariableOp_2Ђ+lstm_71/while/lstm_cell_71/ReadVariableOp_3Ђ/lstm_71/while/lstm_cell_71/split/ReadVariableOpЂ1lstm_71/while/lstm_cell_71/split_1/ReadVariableOpг
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_71/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_71/while/lstm_cell_71/ones_like/ShapeShapelstm_71_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_71/while/lstm_cell_71/ones_like/Shape
*lstm_71/while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_71/while/lstm_cell_71/ones_like/Const№
$lstm_71/while/lstm_cell_71/ones_likeFill3lstm_71/while/lstm_cell_71/ones_like/Shape:output:03lstm_71/while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/ones_like
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_71/while/lstm_cell_71/split/split_dimо
/lstm_71/while/lstm_cell_71/split/ReadVariableOpReadVariableOp:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_71/while/lstm_cell_71/split/ReadVariableOp
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:07lstm_71/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_71/while/lstm_cell_71/splitч
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_71/while/lstm_cell_71/MatMulы
#lstm_71/while/lstm_cell_71/MatMul_1MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_1ы
#lstm_71/while/lstm_cell_71/MatMul_2MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_2ы
#lstm_71/while/lstm_cell_71/MatMul_3MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_3
,lstm_71/while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_71/while/lstm_cell_71/split_1/split_dimр
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp
"lstm_71/while/lstm_cell_71/split_1Split5lstm_71/while/lstm_cell_71/split_1/split_dim:output:09lstm_71/while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_71/while/lstm_cell_71/split_1п
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd+lstm_71/while/lstm_cell_71/MatMul:product:0+lstm_71/while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/while/lstm_cell_71/BiasAddх
$lstm_71/while/lstm_cell_71/BiasAdd_1BiasAdd-lstm_71/while/lstm_cell_71/MatMul_1:product:0+lstm_71/while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_1х
$lstm_71/while/lstm_cell_71/BiasAdd_2BiasAdd-lstm_71/while/lstm_cell_71/MatMul_2:product:0+lstm_71/while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_2х
$lstm_71/while/lstm_cell_71/BiasAdd_3BiasAdd-lstm_71/while/lstm_cell_71/MatMul_3:product:0+lstm_71/while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_3Х
lstm_71/while/lstm_cell_71/mulMullstm_71_while_placeholder_2-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/while/lstm_cell_71/mulЩ
 lstm_71/while/lstm_cell_71/mul_1Mullstm_71_while_placeholder_2-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_1Щ
 lstm_71/while/lstm_cell_71/mul_2Mullstm_71_while_placeholder_2-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_2Щ
 lstm_71/while/lstm_cell_71/mul_3Mullstm_71_while_placeholder_2-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_3Ь
)lstm_71/while/lstm_cell_71/ReadVariableOpReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_71/while/lstm_cell_71/ReadVariableOpБ
.lstm_71/while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_71/while/lstm_cell_71/strided_slice/stackЕ
0lstm_71/while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_71/while/lstm_cell_71/strided_slice/stack_1Е
0lstm_71/while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_71/while/lstm_cell_71/strided_slice/stack_2
(lstm_71/while/lstm_cell_71/strided_sliceStridedSlice1lstm_71/while/lstm_cell_71/ReadVariableOp:value:07lstm_71/while/lstm_cell_71/strided_slice/stack:output:09lstm_71/while/lstm_cell_71/strided_slice/stack_1:output:09lstm_71/while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_71/while/lstm_cell_71/strided_sliceн
#lstm_71/while/lstm_cell_71/MatMul_4MatMul"lstm_71/while/lstm_cell_71/mul:z:01lstm_71/while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_4з
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/BiasAdd:output:0-lstm_71/while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/while/lstm_cell_71/addЉ
"lstm_71/while/lstm_cell_71/SigmoidSigmoid"lstm_71/while/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/while/lstm_cell_71/Sigmoidа
+lstm_71/while/lstm_cell_71/ReadVariableOp_1ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_1Е
0lstm_71/while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_71/while/lstm_cell_71/strided_slice_1/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_1StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_1:value:09lstm_71/while/lstm_cell_71/strided_slice_1/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_1/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_1с
#lstm_71/while/lstm_cell_71/MatMul_5MatMul$lstm_71/while/lstm_cell_71/mul_1:z:03lstm_71/while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_5н
 lstm_71/while/lstm_cell_71/add_1AddV2-lstm_71/while/lstm_cell_71/BiasAdd_1:output:0-lstm_71/while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_1Џ
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/Sigmoid_1Ф
 lstm_71/while/lstm_cell_71/mul_4Mul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_4а
+lstm_71/while/lstm_cell_71/ReadVariableOp_2ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_2Е
0lstm_71/while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_71/while/lstm_cell_71/strided_slice_2/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_2StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_2:value:09lstm_71/while/lstm_cell_71/strided_slice_2/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_2/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_2с
#lstm_71/while/lstm_cell_71/MatMul_6MatMul$lstm_71/while/lstm_cell_71/mul_2:z:03lstm_71/while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_6н
 lstm_71/while/lstm_cell_71/add_2AddV2-lstm_71/while/lstm_cell_71/BiasAdd_2:output:0-lstm_71/while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_2Ђ
lstm_71/while/lstm_cell_71/ReluRelu$lstm_71/while/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_71/while/lstm_cell_71/Reluд
 lstm_71/while/lstm_cell_71/mul_5Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_5Ы
 lstm_71/while/lstm_cell_71/add_3AddV2$lstm_71/while/lstm_cell_71/mul_4:z:0$lstm_71/while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_3а
+lstm_71/while/lstm_cell_71/ReadVariableOp_3ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_3Е
0lstm_71/while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_71/while/lstm_cell_71/strided_slice_3/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_3StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_3:value:09lstm_71/while/lstm_cell_71/strided_slice_3/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_3/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_3с
#lstm_71/while/lstm_cell_71/MatMul_7MatMul$lstm_71/while/lstm_cell_71/mul_3:z:03lstm_71/while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_7н
 lstm_71/while/lstm_cell_71/add_4AddV2-lstm_71/while/lstm_cell_71/BiasAdd_3:output:0-lstm_71/while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_4Џ
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid$lstm_71/while/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/Sigmoid_2І
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_71/while/lstm_cell_71/Relu_1и
 lstm_71/while/lstm_cell_71/mul_6Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_6
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1lstm_71_while_placeholder$lstm_71/while/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_71/while/TensorArrayV2Write/TensorListSetIteml
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add/y
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/addp
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add_1/y
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/add_1
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/IdentityІ
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_1
lstm_71/while/Identity_2Identitylstm_71/while/add:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_2К
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_3­
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_6:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/while/Identity_4­
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_3:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/while/Identity_5
lstm_71/while/NoOpNoOp*^lstm_71/while/lstm_cell_71/ReadVariableOp,^lstm_71/while/lstm_cell_71/ReadVariableOp_1,^lstm_71/while/lstm_cell_71/ReadVariableOp_2,^lstm_71/while/lstm_cell_71/ReadVariableOp_30^lstm_71/while/lstm_cell_71/split/ReadVariableOp2^lstm_71/while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_71/while/NoOp"9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"j
2lstm_71_while_lstm_cell_71_readvariableop_resource4lstm_71_while_lstm_cell_71_readvariableop_resource_0"z
:lstm_71_while_lstm_cell_71_split_1_readvariableop_resource<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0"v
8lstm_71_while_lstm_cell_71_split_readvariableop_resource:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0"Ш
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_71/while/lstm_cell_71/ReadVariableOp)lstm_71/while/lstm_cell_71/ReadVariableOp2Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_1+lstm_71/while/lstm_cell_71/ReadVariableOp_12Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_2+lstm_71/while/lstm_cell_71/ReadVariableOp_22Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_3+lstm_71/while/lstm_cell_71/ReadVariableOp_32b
/lstm_71/while/lstm_cell_71/split/ReadVariableOp/lstm_71/while/lstm_cell_71/split/ReadVariableOp2f
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
сЁ
Ј
D__inference_lstm_71_layer_call_and_return_conditional_losses_2303564

inputs=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2303431*
condR
while_cond_2303430*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_2305562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2305562___redundant_placeholder05
1while_while_cond_2305562___redundant_placeholder15
1while_while_cond_2305562___redundant_placeholder25
1while_while_cond_2305562___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
щ%
ъ
while_body_2302669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_71_2302693_0:	+
while_lstm_cell_71_2302695_0:	/
while_lstm_cell_71_2302697_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_71_2302693:	)
while_lstm_cell_71_2302695:	-
while_lstm_cell_71_2302697:	 Ђ*while/lstm_cell_71/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_2302693_0while_lstm_cell_71_2302695_0while_lstm_cell_71_2302697_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23026552,
*while/lstm_cell_71/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Є
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_71_2302693while_lstm_cell_71_2302693_0":
while_lstm_cell_71_2302695while_lstm_cell_71_2302695_0":
while_lstm_cell_71_2302697while_lstm_cell_71_2302697_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ъ	
Њ
/__inference_sequential_29_layer_call_fn_2304102
input_30
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_29_layer_call_and_return_conditional_losses_23040662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
ј+
Е
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304170
input_30"
lstm_71_2304139:	
lstm_71_2304141:	"
lstm_71_2304143:	 "
dense_86_2304146:  
dense_86_2304148: "
dense_87_2304151: 
dense_87_2304153:
identityЂ dense_86/StatefulPartitionedCallЂ dense_87/StatefulPartitionedCallЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂlstm_71/StatefulPartitionedCallЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЇ
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinput_30lstm_71_2304139lstm_71_2304141lstm_71_2304143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23040022!
lstm_71/StatefulPartitionedCallЙ
 dense_86/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_86_2304146dense_86_2304148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_23035832"
 dense_86/StatefulPartitionedCallК
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_2304151dense_87_2304153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_23036052"
 dense_87/StatefulPartitionedCall
reshape_43/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_43_layer_call_and_return_conditional_losses_23036242
reshape_43/PartitionedCallЯ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_71_2304139*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЏ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_2304153*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mul
IdentityIdentity#reshape_43/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall0^dense_87/bias/Regularizer/Square/ReadVariableOp ^lstm_71/StatefulPartitionedCall>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
Аў
	
"__inference__wrapped_model_2302531
input_30S
@sequential_29_lstm_71_lstm_cell_71_split_readvariableop_resource:	Q
Bsequential_29_lstm_71_lstm_cell_71_split_1_readvariableop_resource:	M
:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource:	 G
5sequential_29_dense_86_matmul_readvariableop_resource:  D
6sequential_29_dense_86_biasadd_readvariableop_resource: G
5sequential_29_dense_87_matmul_readvariableop_resource: D
6sequential_29_dense_87_biasadd_readvariableop_resource:
identityЂ-sequential_29/dense_86/BiasAdd/ReadVariableOpЂ,sequential_29/dense_86/MatMul/ReadVariableOpЂ-sequential_29/dense_87/BiasAdd/ReadVariableOpЂ,sequential_29/dense_87/MatMul/ReadVariableOpЂ1sequential_29/lstm_71/lstm_cell_71/ReadVariableOpЂ3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_1Ђ3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_2Ђ3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_3Ђ7sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOpЂ9sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOpЂsequential_29/lstm_71/whiler
sequential_29/lstm_71/ShapeShapeinput_30*
T0*
_output_shapes
:2
sequential_29/lstm_71/Shape 
)sequential_29/lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_29/lstm_71/strided_slice/stackЄ
+sequential_29/lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_29/lstm_71/strided_slice/stack_1Є
+sequential_29/lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_29/lstm_71/strided_slice/stack_2ц
#sequential_29/lstm_71/strided_sliceStridedSlice$sequential_29/lstm_71/Shape:output:02sequential_29/lstm_71/strided_slice/stack:output:04sequential_29/lstm_71/strided_slice/stack_1:output:04sequential_29/lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_29/lstm_71/strided_slice
!sequential_29/lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_29/lstm_71/zeros/mul/yФ
sequential_29/lstm_71/zeros/mulMul,sequential_29/lstm_71/strided_slice:output:0*sequential_29/lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_29/lstm_71/zeros/mul
"sequential_29/lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_29/lstm_71/zeros/Less/yП
 sequential_29/lstm_71/zeros/LessLess#sequential_29/lstm_71/zeros/mul:z:0+sequential_29/lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_29/lstm_71/zeros/Less
$sequential_29/lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_29/lstm_71/zeros/packed/1л
"sequential_29/lstm_71/zeros/packedPack,sequential_29/lstm_71/strided_slice:output:0-sequential_29/lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_29/lstm_71/zeros/packed
!sequential_29/lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_29/lstm_71/zeros/ConstЭ
sequential_29/lstm_71/zerosFill+sequential_29/lstm_71/zeros/packed:output:0*sequential_29/lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_29/lstm_71/zeros
#sequential_29/lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_29/lstm_71/zeros_1/mul/yЪ
!sequential_29/lstm_71/zeros_1/mulMul,sequential_29/lstm_71/strided_slice:output:0,sequential_29/lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_29/lstm_71/zeros_1/mul
$sequential_29/lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_29/lstm_71/zeros_1/Less/yЧ
"sequential_29/lstm_71/zeros_1/LessLess%sequential_29/lstm_71/zeros_1/mul:z:0-sequential_29/lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_29/lstm_71/zeros_1/Less
&sequential_29/lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_29/lstm_71/zeros_1/packed/1с
$sequential_29/lstm_71/zeros_1/packedPack,sequential_29/lstm_71/strided_slice:output:0/sequential_29/lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_29/lstm_71/zeros_1/packed
#sequential_29/lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_29/lstm_71/zeros_1/Constе
sequential_29/lstm_71/zeros_1Fill-sequential_29/lstm_71/zeros_1/packed:output:0,sequential_29/lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_29/lstm_71/zeros_1Ё
$sequential_29/lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_29/lstm_71/transpose/permО
sequential_29/lstm_71/transpose	Transposeinput_30-sequential_29/lstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_29/lstm_71/transpose
sequential_29/lstm_71/Shape_1Shape#sequential_29/lstm_71/transpose:y:0*
T0*
_output_shapes
:2
sequential_29/lstm_71/Shape_1Є
+sequential_29/lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_29/lstm_71/strided_slice_1/stackЈ
-sequential_29/lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_29/lstm_71/strided_slice_1/stack_1Ј
-sequential_29/lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_29/lstm_71/strided_slice_1/stack_2ђ
%sequential_29/lstm_71/strided_slice_1StridedSlice&sequential_29/lstm_71/Shape_1:output:04sequential_29/lstm_71/strided_slice_1/stack:output:06sequential_29/lstm_71/strided_slice_1/stack_1:output:06sequential_29/lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_29/lstm_71/strided_slice_1Б
1sequential_29/lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ23
1sequential_29/lstm_71/TensorArrayV2/element_shape
#sequential_29/lstm_71/TensorArrayV2TensorListReserve:sequential_29/lstm_71/TensorArrayV2/element_shape:output:0.sequential_29/lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_29/lstm_71/TensorArrayV2ы
Ksequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2M
Ksequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeа
=sequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_29/lstm_71/transpose:y:0Tsequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensorЄ
+sequential_29/lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_29/lstm_71/strided_slice_2/stackЈ
-sequential_29/lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_29/lstm_71/strided_slice_2/stack_1Ј
-sequential_29/lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_29/lstm_71/strided_slice_2/stack_2
%sequential_29/lstm_71/strided_slice_2StridedSlice#sequential_29/lstm_71/transpose:y:04sequential_29/lstm_71/strided_slice_2/stack:output:06sequential_29/lstm_71/strided_slice_2/stack_1:output:06sequential_29/lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2'
%sequential_29/lstm_71/strided_slice_2М
2sequential_29/lstm_71/lstm_cell_71/ones_like/ShapeShape$sequential_29/lstm_71/zeros:output:0*
T0*
_output_shapes
:24
2sequential_29/lstm_71/lstm_cell_71/ones_like/Shape­
2sequential_29/lstm_71/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_29/lstm_71/lstm_cell_71/ones_like/Const
,sequential_29/lstm_71/lstm_cell_71/ones_likeFill;sequential_29/lstm_71/lstm_cell_71/ones_like/Shape:output:0;sequential_29/lstm_71/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/ones_likeЊ
2sequential_29/lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_29/lstm_71/lstm_cell_71/split/split_dimє
7sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOpReadVariableOp@sequential_29_lstm_71_lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype029
7sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOpГ
(sequential_29/lstm_71/lstm_cell_71/splitSplit;sequential_29/lstm_71/lstm_cell_71/split/split_dim:output:0?sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2*
(sequential_29/lstm_71/lstm_cell_71/splitѕ
)sequential_29/lstm_71/lstm_cell_71/MatMulMatMul.sequential_29/lstm_71/strided_slice_2:output:01sequential_29/lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_29/lstm_71/lstm_cell_71/MatMulљ
+sequential_29/lstm_71/lstm_cell_71/MatMul_1MatMul.sequential_29/lstm_71/strided_slice_2:output:01sequential_29/lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_1љ
+sequential_29/lstm_71/lstm_cell_71/MatMul_2MatMul.sequential_29/lstm_71/strided_slice_2:output:01sequential_29/lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_2љ
+sequential_29/lstm_71/lstm_cell_71/MatMul_3MatMul.sequential_29/lstm_71/strided_slice_2:output:01sequential_29/lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_3Ў
4sequential_29/lstm_71/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_29/lstm_71/lstm_cell_71/split_1/split_dimі
9sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOpReadVariableOpBsequential_29_lstm_71_lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOpЋ
*sequential_29/lstm_71/lstm_cell_71/split_1Split=sequential_29/lstm_71/lstm_cell_71/split_1/split_dim:output:0Asequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2,
*sequential_29/lstm_71/lstm_cell_71/split_1џ
*sequential_29/lstm_71/lstm_cell_71/BiasAddBiasAdd3sequential_29/lstm_71/lstm_cell_71/MatMul:product:03sequential_29/lstm_71/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_29/lstm_71/lstm_cell_71/BiasAdd
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_1BiasAdd5sequential_29/lstm_71/lstm_cell_71/MatMul_1:product:03sequential_29/lstm_71/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_1
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_2BiasAdd5sequential_29/lstm_71/lstm_cell_71/MatMul_2:product:03sequential_29/lstm_71/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_2
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_3BiasAdd5sequential_29/lstm_71/lstm_cell_71/MatMul_3:product:03sequential_29/lstm_71/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/BiasAdd_3ц
&sequential_29/lstm_71/lstm_cell_71/mulMul$sequential_29/lstm_71/zeros:output:05sequential_29/lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_29/lstm_71/lstm_cell_71/mulъ
(sequential_29/lstm_71/lstm_cell_71/mul_1Mul$sequential_29/lstm_71/zeros:output:05sequential_29/lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_1ъ
(sequential_29/lstm_71/lstm_cell_71/mul_2Mul$sequential_29/lstm_71/zeros:output:05sequential_29/lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_2ъ
(sequential_29/lstm_71/lstm_cell_71/mul_3Mul$sequential_29/lstm_71/zeros:output:05sequential_29/lstm_71/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_3т
1sequential_29/lstm_71/lstm_cell_71/ReadVariableOpReadVariableOp:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype023
1sequential_29/lstm_71/lstm_cell_71/ReadVariableOpС
6sequential_29/lstm_71/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_29/lstm_71/lstm_cell_71/strided_slice/stackХ
8sequential_29/lstm_71/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_29/lstm_71/lstm_cell_71/strided_slice/stack_1Х
8sequential_29/lstm_71/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_29/lstm_71/lstm_cell_71/strided_slice/stack_2Ю
0sequential_29/lstm_71/lstm_cell_71/strided_sliceStridedSlice9sequential_29/lstm_71/lstm_cell_71/ReadVariableOp:value:0?sequential_29/lstm_71/lstm_cell_71/strided_slice/stack:output:0Asequential_29/lstm_71/lstm_cell_71/strided_slice/stack_1:output:0Asequential_29/lstm_71/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_29/lstm_71/lstm_cell_71/strided_slice§
+sequential_29/lstm_71/lstm_cell_71/MatMul_4MatMul*sequential_29/lstm_71/lstm_cell_71/mul:z:09sequential_29/lstm_71/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_4ї
&sequential_29/lstm_71/lstm_cell_71/addAddV23sequential_29/lstm_71/lstm_cell_71/BiasAdd:output:05sequential_29/lstm_71/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_29/lstm_71/lstm_cell_71/addС
*sequential_29/lstm_71/lstm_cell_71/SigmoidSigmoid*sequential_29/lstm_71/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_29/lstm_71/lstm_cell_71/Sigmoidц
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_1ReadVariableOp:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_1Х
8sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stackЩ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_1Щ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_2к
2sequential_29/lstm_71/lstm_cell_71/strided_slice_1StridedSlice;sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_1:value:0Asequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_1:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_29/lstm_71/lstm_cell_71/strided_slice_1
+sequential_29/lstm_71/lstm_cell_71/MatMul_5MatMul,sequential_29/lstm_71/lstm_cell_71/mul_1:z:0;sequential_29/lstm_71/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_5§
(sequential_29/lstm_71/lstm_cell_71/add_1AddV25sequential_29/lstm_71/lstm_cell_71/BiasAdd_1:output:05sequential_29/lstm_71/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/add_1Ч
,sequential_29/lstm_71/lstm_cell_71/Sigmoid_1Sigmoid,sequential_29/lstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/Sigmoid_1ч
(sequential_29/lstm_71/lstm_cell_71/mul_4Mul0sequential_29/lstm_71/lstm_cell_71/Sigmoid_1:y:0&sequential_29/lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_4ц
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_2ReadVariableOp:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_2Х
8sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stackЩ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_1Щ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_2к
2sequential_29/lstm_71/lstm_cell_71/strided_slice_2StridedSlice;sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_2:value:0Asequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_1:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_29/lstm_71/lstm_cell_71/strided_slice_2
+sequential_29/lstm_71/lstm_cell_71/MatMul_6MatMul,sequential_29/lstm_71/lstm_cell_71/mul_2:z:0;sequential_29/lstm_71/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_6§
(sequential_29/lstm_71/lstm_cell_71/add_2AddV25sequential_29/lstm_71/lstm_cell_71/BiasAdd_2:output:05sequential_29/lstm_71/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/add_2К
'sequential_29/lstm_71/lstm_cell_71/ReluRelu,sequential_29/lstm_71/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_29/lstm_71/lstm_cell_71/Reluє
(sequential_29/lstm_71/lstm_cell_71/mul_5Mul.sequential_29/lstm_71/lstm_cell_71/Sigmoid:y:05sequential_29/lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_5ы
(sequential_29/lstm_71/lstm_cell_71/add_3AddV2,sequential_29/lstm_71/lstm_cell_71/mul_4:z:0,sequential_29/lstm_71/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/add_3ц
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_3ReadVariableOp:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_3Х
8sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2:
8sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stackЩ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_1Щ
:sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_2к
2sequential_29/lstm_71/lstm_cell_71/strided_slice_3StridedSlice;sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_3:value:0Asequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_1:output:0Csequential_29/lstm_71/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_29/lstm_71/lstm_cell_71/strided_slice_3
+sequential_29/lstm_71/lstm_cell_71/MatMul_7MatMul,sequential_29/lstm_71/lstm_cell_71/mul_3:z:0;sequential_29/lstm_71/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_29/lstm_71/lstm_cell_71/MatMul_7§
(sequential_29/lstm_71/lstm_cell_71/add_4AddV25sequential_29/lstm_71/lstm_cell_71/BiasAdd_3:output:05sequential_29/lstm_71/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/add_4Ч
,sequential_29/lstm_71/lstm_cell_71/Sigmoid_2Sigmoid,sequential_29/lstm_71/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_29/lstm_71/lstm_cell_71/Sigmoid_2О
)sequential_29/lstm_71/lstm_cell_71/Relu_1Relu,sequential_29/lstm_71/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_29/lstm_71/lstm_cell_71/Relu_1ј
(sequential_29/lstm_71/lstm_cell_71/mul_6Mul0sequential_29/lstm_71/lstm_cell_71/Sigmoid_2:y:07sequential_29/lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_29/lstm_71/lstm_cell_71/mul_6Л
3sequential_29/lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    25
3sequential_29/lstm_71/TensorArrayV2_1/element_shape
%sequential_29/lstm_71/TensorArrayV2_1TensorListReserve<sequential_29/lstm_71/TensorArrayV2_1/element_shape:output:0.sequential_29/lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_29/lstm_71/TensorArrayV2_1z
sequential_29/lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_29/lstm_71/timeЋ
.sequential_29/lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.sequential_29/lstm_71/while/maximum_iterations
(sequential_29/lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_29/lstm_71/while/loop_counterЭ
sequential_29/lstm_71/whileWhile1sequential_29/lstm_71/while/loop_counter:output:07sequential_29/lstm_71/while/maximum_iterations:output:0#sequential_29/lstm_71/time:output:0.sequential_29/lstm_71/TensorArrayV2_1:handle:0$sequential_29/lstm_71/zeros:output:0&sequential_29/lstm_71/zeros_1:output:0.sequential_29/lstm_71/strided_slice_1:output:0Msequential_29/lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_29_lstm_71_lstm_cell_71_split_readvariableop_resourceBsequential_29_lstm_71_lstm_cell_71_split_1_readvariableop_resource:sequential_29_lstm_71_lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_29_lstm_71_while_body_2302382*4
cond,R*
(sequential_29_lstm_71_while_cond_2302381*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_29/lstm_71/whileс
Fsequential_29/lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2H
Fsequential_29/lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeР
8sequential_29/lstm_71/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_29/lstm_71/while:output:3Osequential_29/lstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02:
8sequential_29/lstm_71/TensorArrayV2Stack/TensorListStack­
+sequential_29/lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2-
+sequential_29/lstm_71/strided_slice_3/stackЈ
-sequential_29/lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_29/lstm_71/strided_slice_3/stack_1Ј
-sequential_29/lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_29/lstm_71/strided_slice_3/stack_2
%sequential_29/lstm_71/strided_slice_3StridedSliceAsequential_29/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:04sequential_29/lstm_71/strided_slice_3/stack:output:06sequential_29/lstm_71/strided_slice_3/stack_1:output:06sequential_29/lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2'
%sequential_29/lstm_71/strided_slice_3Ѕ
&sequential_29/lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_29/lstm_71/transpose_1/perm§
!sequential_29/lstm_71/transpose_1	TransposeAsequential_29/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_29/lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2#
!sequential_29/lstm_71/transpose_1
sequential_29/lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_29/lstm_71/runtimeв
,sequential_29/dense_86/MatMul/ReadVariableOpReadVariableOp5sequential_29_dense_86_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02.
,sequential_29/dense_86/MatMul/ReadVariableOpр
sequential_29/dense_86/MatMulMatMul.sequential_29/lstm_71/strided_slice_3:output:04sequential_29/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_29/dense_86/MatMulб
-sequential_29/dense_86/BiasAdd/ReadVariableOpReadVariableOp6sequential_29_dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_29/dense_86/BiasAdd/ReadVariableOpн
sequential_29/dense_86/BiasAddBiasAdd'sequential_29/dense_86/MatMul:product:05sequential_29/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
sequential_29/dense_86/BiasAdd
sequential_29/dense_86/ReluRelu'sequential_29/dense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_29/dense_86/Reluв
,sequential_29/dense_87/MatMul/ReadVariableOpReadVariableOp5sequential_29_dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_29/dense_87/MatMul/ReadVariableOpл
sequential_29/dense_87/MatMulMatMul)sequential_29/dense_86/Relu:activations:04sequential_29/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_29/dense_87/MatMulб
-sequential_29/dense_87/BiasAdd/ReadVariableOpReadVariableOp6sequential_29_dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_29/dense_87/BiasAdd/ReadVariableOpн
sequential_29/dense_87/BiasAddBiasAdd'sequential_29/dense_87/MatMul:product:05sequential_29/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_29/dense_87/BiasAdd
sequential_29/reshape_43/ShapeShape'sequential_29/dense_87/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_29/reshape_43/ShapeІ
,sequential_29/reshape_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_29/reshape_43/strided_slice/stackЊ
.sequential_29/reshape_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_29/reshape_43/strided_slice/stack_1Њ
.sequential_29/reshape_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_29/reshape_43/strided_slice/stack_2ј
&sequential_29/reshape_43/strided_sliceStridedSlice'sequential_29/reshape_43/Shape:output:05sequential_29/reshape_43/strided_slice/stack:output:07sequential_29/reshape_43/strided_slice/stack_1:output:07sequential_29/reshape_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_29/reshape_43/strided_slice
(sequential_29/reshape_43/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_29/reshape_43/Reshape/shape/1
(sequential_29/reshape_43/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_29/reshape_43/Reshape/shape/2
&sequential_29/reshape_43/Reshape/shapePack/sequential_29/reshape_43/strided_slice:output:01sequential_29/reshape_43/Reshape/shape/1:output:01sequential_29/reshape_43/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_29/reshape_43/Reshape/shapeп
 sequential_29/reshape_43/ReshapeReshape'sequential_29/dense_87/BiasAdd:output:0/sequential_29/reshape_43/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_29/reshape_43/Reshape
IdentityIdentity)sequential_29/reshape_43/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityі
NoOpNoOp.^sequential_29/dense_86/BiasAdd/ReadVariableOp-^sequential_29/dense_86/MatMul/ReadVariableOp.^sequential_29/dense_87/BiasAdd/ReadVariableOp-^sequential_29/dense_87/MatMul/ReadVariableOp2^sequential_29/lstm_71/lstm_cell_71/ReadVariableOp4^sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_14^sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_24^sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_38^sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOp:^sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOp^sequential_29/lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2^
-sequential_29/dense_86/BiasAdd/ReadVariableOp-sequential_29/dense_86/BiasAdd/ReadVariableOp2\
,sequential_29/dense_86/MatMul/ReadVariableOp,sequential_29/dense_86/MatMul/ReadVariableOp2^
-sequential_29/dense_87/BiasAdd/ReadVariableOp-sequential_29/dense_87/BiasAdd/ReadVariableOp2\
,sequential_29/dense_87/MatMul/ReadVariableOp,sequential_29/dense_87/MatMul/ReadVariableOp2f
1sequential_29/lstm_71/lstm_cell_71/ReadVariableOp1sequential_29/lstm_71/lstm_cell_71/ReadVariableOp2j
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_13sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_12j
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_23sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_22j
3sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_33sequential_29/lstm_71/lstm_cell_71/ReadVariableOp_32r
7sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOp7sequential_29/lstm_71/lstm_cell_71/split/ReadVariableOp2v
9sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOp9sequential_29/lstm_71/lstm_cell_71/split_1/ReadVariableOp2:
sequential_29/lstm_71/whilesequential_29/lstm_71/while:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
Ј
Ѕ	
while_body_2303431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Ѕ
while/lstm_cell_71/mulMulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЉ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Љ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Љ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Ђ|

#__inference__traced_restore_2306529
file_prefix2
 assignvariableop_dense_86_kernel:  .
 assignvariableop_1_dense_86_bias: 4
"assignvariableop_2_dense_87_kernel: .
 assignvariableop_3_dense_87_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_71_lstm_cell_71_kernel:	L
9assignvariableop_10_lstm_71_lstm_cell_71_recurrent_kernel:	 <
-assignvariableop_11_lstm_71_lstm_cell_71_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_86_kernel_m:  6
(assignvariableop_15_adam_dense_86_bias_m: <
*assignvariableop_16_adam_dense_87_kernel_m: 6
(assignvariableop_17_adam_dense_87_bias_m:I
6assignvariableop_18_adam_lstm_71_lstm_cell_71_kernel_m:	S
@assignvariableop_19_adam_lstm_71_lstm_cell_71_recurrent_kernel_m:	 C
4assignvariableop_20_adam_lstm_71_lstm_cell_71_bias_m:	<
*assignvariableop_21_adam_dense_86_kernel_v:  6
(assignvariableop_22_adam_dense_86_bias_v: <
*assignvariableop_23_adam_dense_87_kernel_v: 6
(assignvariableop_24_adam_dense_87_bias_v:I
6assignvariableop_25_adam_lstm_71_lstm_cell_71_kernel_v:	S
@assignvariableop_26_adam_lstm_71_lstm_cell_71_recurrent_kernel_v:	 C
4assignvariableop_27_adam_lstm_71_lstm_cell_71_bias_v:	
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
valueиBеB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesН
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_86_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_86_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_87_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_87_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4Ё
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ѓ
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Њ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Г
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_71_lstm_cell_71_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10С
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_71_lstm_cell_71_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_71_lstm_cell_71_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ё
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14В
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_86_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15А
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_86_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_87_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_87_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18О
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_71_lstm_cell_71_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ш
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_71_lstm_cell_71_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20М
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_71_lstm_cell_71_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_86_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_86_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_87_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_87_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25О
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_71_lstm_cell_71_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ш
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_71_lstm_cell_71_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27М
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_71_lstm_cell_71_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28f
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_29Ў
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
щ%
ъ
while_body_2302966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_71_2302990_0:	+
while_lstm_cell_71_2302992_0:	/
while_lstm_cell_71_2302994_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_71_2302990:	)
while_lstm_cell_71_2302992:	-
while_lstm_cell_71_2302994:	 Ђ*while/lstm_cell_71/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_2302990_0while_lstm_cell_71_2302992_0while_lstm_cell_71_2302994_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23028882,
*while/lstm_cell_71/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Є
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_71_2302990while_lstm_cell_71_2302990_0":
while_lstm_cell_71_2302992while_lstm_cell_71_2302992_0":
while_lstm_cell_71_2302994while_lstm_cell_71_2302994_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
к
Ш
while_cond_2303836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2303836___redundant_placeholder05
1while_while_cond_2303836___redundant_placeholder15
1while_while_cond_2303836___redundant_placeholder25
1while_while_cond_2303836___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
пR
ь
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306204

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6н
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
 

J__inference_sequential_29_layer_call_and_return_conditional_losses_2304853

inputsE
2lstm_71_lstm_cell_71_split_readvariableop_resource:	C
4lstm_71_lstm_cell_71_split_1_readvariableop_resource:	?
,lstm_71_lstm_cell_71_readvariableop_resource:	 9
'dense_86_matmul_readvariableop_resource:  6
(dense_86_biasadd_readvariableop_resource: 9
'dense_87_matmul_readvariableop_resource: 6
(dense_87_biasadd_readvariableop_resource:
identityЂdense_86/BiasAdd/ReadVariableOpЂdense_86/MatMul/ReadVariableOpЂdense_87/BiasAdd/ReadVariableOpЂdense_87/MatMul/ReadVariableOpЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂ#lstm_71/lstm_cell_71/ReadVariableOpЂ%lstm_71/lstm_cell_71/ReadVariableOp_1Ђ%lstm_71/lstm_cell_71/ReadVariableOp_2Ђ%lstm_71/lstm_cell_71/ReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_71/lstm_cell_71/split/ReadVariableOpЂ+lstm_71/lstm_cell_71/split_1/ReadVariableOpЂlstm_71/whileT
lstm_71/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_71/Shape
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice/stack
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_1
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_2
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slicel
lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros/mul/y
lstm_71/zeros/mulMullstm_71/strided_slice:output:0lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/mulo
lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_71/zeros/Less/y
lstm_71/zeros/LessLesslstm_71/zeros/mul:z:0lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/Lessr
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros/packed/1Ѓ
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros/packedo
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros/Const
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/zerosp
lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros_1/mul/y
lstm_71/zeros_1/mulMullstm_71/strided_slice:output:0lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/muls
lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_71/zeros_1/Less/y
lstm_71/zeros_1/LessLesslstm_71/zeros_1/mul:z:0lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/Lessv
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/zeros_1/packed/1Љ
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros_1/packeds
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros_1/Const
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/zeros_1
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose/perm
lstm_71/transpose	Transposeinputslstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_71/transposeg
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:2
lstm_71/Shape_1
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_1/stack
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_1
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_2
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slice_1
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_71/TensorArrayV2/element_shapeв
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2Я
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_71/TensorArrayUnstack/TensorListFromTensor
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_2/stack
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_1
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_2Ќ
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_71/strided_slice_2
$lstm_71/lstm_cell_71/ones_like/ShapeShapelstm_71/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_71/lstm_cell_71/ones_like/Shape
$lstm_71/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_71/lstm_cell_71/ones_like/Constи
lstm_71/lstm_cell_71/ones_likeFill-lstm_71/lstm_cell_71/ones_like/Shape:output:0-lstm_71/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/ones_like
"lstm_71/lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"lstm_71/lstm_cell_71/dropout/Constг
 lstm_71/lstm_cell_71/dropout/MulMul'lstm_71/lstm_cell_71/ones_like:output:0+lstm_71/lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/lstm_cell_71/dropout/Mul
"lstm_71/lstm_cell_71/dropout/ShapeShape'lstm_71/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_71/lstm_cell_71/dropout/Shape
9lstm_71/lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform+lstm_71/lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2їћў2;
9lstm_71/lstm_cell_71/dropout/random_uniform/RandomUniform
+lstm_71/lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+lstm_71/lstm_cell_71/dropout/GreaterEqual/y
)lstm_71/lstm_cell_71/dropout/GreaterEqualGreaterEqualBlstm_71/lstm_cell_71/dropout/random_uniform/RandomUniform:output:04lstm_71/lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_71/lstm_cell_71/dropout/GreaterEqualО
!lstm_71/lstm_cell_71/dropout/CastCast-lstm_71/lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_71/lstm_cell_71/dropout/CastЮ
"lstm_71/lstm_cell_71/dropout/Mul_1Mul$lstm_71/lstm_cell_71/dropout/Mul:z:0%lstm_71/lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/lstm_cell_71/dropout/Mul_1
$lstm_71/lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_71/lstm_cell_71/dropout_1/Constй
"lstm_71/lstm_cell_71/dropout_1/MulMul'lstm_71/lstm_cell_71/ones_like:output:0-lstm_71/lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/lstm_cell_71/dropout_1/MulЃ
$lstm_71/lstm_cell_71/dropout_1/ShapeShape'lstm_71/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_71/lstm_cell_71/dropout_1/Shape
;lstm_71/lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_71/lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2З2=
;lstm_71/lstm_cell_71/dropout_1/random_uniform/RandomUniformЃ
-lstm_71/lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_71/lstm_cell_71/dropout_1/GreaterEqual/y
+lstm_71/lstm_cell_71/dropout_1/GreaterEqualGreaterEqualDlstm_71/lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:06lstm_71/lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_71/lstm_cell_71/dropout_1/GreaterEqualФ
#lstm_71/lstm_cell_71/dropout_1/CastCast/lstm_71/lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/lstm_cell_71/dropout_1/Castж
$lstm_71/lstm_cell_71/dropout_1/Mul_1Mul&lstm_71/lstm_cell_71/dropout_1/Mul:z:0'lstm_71/lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/lstm_cell_71/dropout_1/Mul_1
$lstm_71/lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_71/lstm_cell_71/dropout_2/Constй
"lstm_71/lstm_cell_71/dropout_2/MulMul'lstm_71/lstm_cell_71/ones_like:output:0-lstm_71/lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/lstm_cell_71/dropout_2/MulЃ
$lstm_71/lstm_cell_71/dropout_2/ShapeShape'lstm_71/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_71/lstm_cell_71/dropout_2/Shape
;lstm_71/lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_71/lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ю]2=
;lstm_71/lstm_cell_71/dropout_2/random_uniform/RandomUniformЃ
-lstm_71/lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_71/lstm_cell_71/dropout_2/GreaterEqual/y
+lstm_71/lstm_cell_71/dropout_2/GreaterEqualGreaterEqualDlstm_71/lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:06lstm_71/lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_71/lstm_cell_71/dropout_2/GreaterEqualФ
#lstm_71/lstm_cell_71/dropout_2/CastCast/lstm_71/lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/lstm_cell_71/dropout_2/Castж
$lstm_71/lstm_cell_71/dropout_2/Mul_1Mul&lstm_71/lstm_cell_71/dropout_2/Mul:z:0'lstm_71/lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/lstm_cell_71/dropout_2/Mul_1
$lstm_71/lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_71/lstm_cell_71/dropout_3/Constй
"lstm_71/lstm_cell_71/dropout_3/MulMul'lstm_71/lstm_cell_71/ones_like:output:0-lstm_71/lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/lstm_cell_71/dropout_3/MulЃ
$lstm_71/lstm_cell_71/dropout_3/ShapeShape'lstm_71/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_71/lstm_cell_71/dropout_3/Shape
;lstm_71/lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_71/lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2љАк2=
;lstm_71/lstm_cell_71/dropout_3/random_uniform/RandomUniformЃ
-lstm_71/lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_71/lstm_cell_71/dropout_3/GreaterEqual/y
+lstm_71/lstm_cell_71/dropout_3/GreaterEqualGreaterEqualDlstm_71/lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:06lstm_71/lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_71/lstm_cell_71/dropout_3/GreaterEqualФ
#lstm_71/lstm_cell_71/dropout_3/CastCast/lstm_71/lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/lstm_cell_71/dropout_3/Castж
$lstm_71/lstm_cell_71/dropout_3/Mul_1Mul&lstm_71/lstm_cell_71/dropout_3/Mul:z:0'lstm_71/lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/lstm_cell_71/dropout_3/Mul_1
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_71/lstm_cell_71/split/split_dimЪ
)lstm_71/lstm_cell_71/split/ReadVariableOpReadVariableOp2lstm_71_lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_71/lstm_cell_71/split/ReadVariableOpћ
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:01lstm_71/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_71/lstm_cell_71/splitН
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMulС
lstm_71/lstm_cell_71/MatMul_1MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_1С
lstm_71/lstm_cell_71/MatMul_2MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_2С
lstm_71/lstm_cell_71/MatMul_3MatMul lstm_71/strided_slice_2:output:0#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_3
&lstm_71/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_71/lstm_cell_71/split_1/split_dimЬ
+lstm_71/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_71/lstm_cell_71/split_1/ReadVariableOpѓ
lstm_71/lstm_cell_71/split_1Split/lstm_71/lstm_cell_71/split_1/split_dim:output:03lstm_71/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_71/lstm_cell_71/split_1Ч
lstm_71/lstm_cell_71/BiasAddBiasAdd%lstm_71/lstm_cell_71/MatMul:product:0%lstm_71/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/BiasAddЭ
lstm_71/lstm_cell_71/BiasAdd_1BiasAdd'lstm_71/lstm_cell_71/MatMul_1:product:0%lstm_71/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_1Э
lstm_71/lstm_cell_71/BiasAdd_2BiasAdd'lstm_71/lstm_cell_71/MatMul_2:product:0%lstm_71/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_2Э
lstm_71/lstm_cell_71/BiasAdd_3BiasAdd'lstm_71/lstm_cell_71/MatMul_3:product:0%lstm_71/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/BiasAdd_3­
lstm_71/lstm_cell_71/mulMullstm_71/zeros:output:0&lstm_71/lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mulГ
lstm_71/lstm_cell_71/mul_1Mullstm_71/zeros:output:0(lstm_71/lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_1Г
lstm_71/lstm_cell_71/mul_2Mullstm_71/zeros:output:0(lstm_71/lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_2Г
lstm_71/lstm_cell_71/mul_3Mullstm_71/zeros:output:0(lstm_71/lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_3И
#lstm_71/lstm_cell_71/ReadVariableOpReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_71/lstm_cell_71/ReadVariableOpЅ
(lstm_71/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_71/lstm_cell_71/strided_slice/stackЉ
*lstm_71/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_71/lstm_cell_71/strided_slice/stack_1Љ
*lstm_71/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_71/lstm_cell_71/strided_slice/stack_2њ
"lstm_71/lstm_cell_71/strided_sliceStridedSlice+lstm_71/lstm_cell_71/ReadVariableOp:value:01lstm_71/lstm_cell_71/strided_slice/stack:output:03lstm_71/lstm_cell_71/strided_slice/stack_1:output:03lstm_71/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_71/lstm_cell_71/strided_sliceХ
lstm_71/lstm_cell_71/MatMul_4MatMullstm_71/lstm_cell_71/mul:z:0+lstm_71/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_4П
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/BiasAdd:output:0'lstm_71/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add
lstm_71/lstm_cell_71/SigmoidSigmoidlstm_71/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/SigmoidМ
%lstm_71/lstm_cell_71/ReadVariableOp_1ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_1Љ
*lstm_71/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_71/lstm_cell_71/strided_slice_1/stack­
,lstm_71/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_71/lstm_cell_71/strided_slice_1/stack_1­
,lstm_71/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_1/stack_2
$lstm_71/lstm_cell_71/strided_slice_1StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_1:value:03lstm_71/lstm_cell_71/strided_slice_1/stack:output:05lstm_71/lstm_cell_71/strided_slice_1/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_1Щ
lstm_71/lstm_cell_71/MatMul_5MatMullstm_71/lstm_cell_71/mul_1:z:0-lstm_71/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_5Х
lstm_71/lstm_cell_71/add_1AddV2'lstm_71/lstm_cell_71/BiasAdd_1:output:0'lstm_71/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_1
lstm_71/lstm_cell_71/Sigmoid_1Sigmoidlstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/Sigmoid_1Џ
lstm_71/lstm_cell_71/mul_4Mul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_4М
%lstm_71/lstm_cell_71/ReadVariableOp_2ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_2Љ
*lstm_71/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_71/lstm_cell_71/strided_slice_2/stack­
,lstm_71/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_71/lstm_cell_71/strided_slice_2/stack_1­
,lstm_71/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_2/stack_2
$lstm_71/lstm_cell_71/strided_slice_2StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_2:value:03lstm_71/lstm_cell_71/strided_slice_2/stack:output:05lstm_71/lstm_cell_71/strided_slice_2/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_2Щ
lstm_71/lstm_cell_71/MatMul_6MatMullstm_71/lstm_cell_71/mul_2:z:0-lstm_71/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_6Х
lstm_71/lstm_cell_71/add_2AddV2'lstm_71/lstm_cell_71/BiasAdd_2:output:0'lstm_71/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_2
lstm_71/lstm_cell_71/ReluRelulstm_71/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/ReluМ
lstm_71/lstm_cell_71/mul_5Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_5Г
lstm_71/lstm_cell_71/add_3AddV2lstm_71/lstm_cell_71/mul_4:z:0lstm_71/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_3М
%lstm_71/lstm_cell_71/ReadVariableOp_3ReadVariableOp,lstm_71_lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_71/lstm_cell_71/ReadVariableOp_3Љ
*lstm_71/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_71/lstm_cell_71/strided_slice_3/stack­
,lstm_71/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_71/lstm_cell_71/strided_slice_3/stack_1­
,lstm_71/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_71/lstm_cell_71/strided_slice_3/stack_2
$lstm_71/lstm_cell_71/strided_slice_3StridedSlice-lstm_71/lstm_cell_71/ReadVariableOp_3:value:03lstm_71/lstm_cell_71/strided_slice_3/stack:output:05lstm_71/lstm_cell_71/strided_slice_3/stack_1:output:05lstm_71/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_71/lstm_cell_71/strided_slice_3Щ
lstm_71/lstm_cell_71/MatMul_7MatMullstm_71/lstm_cell_71/mul_3:z:0-lstm_71/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/MatMul_7Х
lstm_71/lstm_cell_71/add_4AddV2'lstm_71/lstm_cell_71/BiasAdd_3:output:0'lstm_71/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/add_4
lstm_71/lstm_cell_71/Sigmoid_2Sigmoidlstm_71/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/lstm_cell_71/Sigmoid_2
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/Relu_1Р
lstm_71/lstm_cell_71/mul_6Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/lstm_cell_71/mul_6
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_71/TensorArrayV2_1/element_shapeи
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2_1^
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/time
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_71/while/maximum_iterationsz
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/while/loop_counterћ
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_71_lstm_cell_71_split_readvariableop_resource4lstm_71_lstm_cell_71_split_1_readvariableop_resource,lstm_71_lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_71_while_body_2304660*&
condR
lstm_71_while_cond_2304659*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_71/whileХ
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_71/TensorArrayV2Stack/TensorListStack
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_71/strided_slice_3/stack
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_71/strided_slice_3/stack_1
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_3/stack_2Ъ
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_71/strided_slice_3
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose_1/permХ
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_71/transpose_1v
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/runtimeЈ
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_86/MatMul/ReadVariableOpЈ
dense_86/MatMulMatMul lstm_71/strided_slice_3:output:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/MatMulЇ
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_86/BiasAdd/ReadVariableOpЅ
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_86/ReluЈ
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_87/MatMul/ReadVariableOpЃ
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_87/MatMulЇ
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOpЅ
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_87/BiasAddm
reshape_43/ShapeShapedense_87/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_43/Shape
reshape_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_43/strided_slice/stack
 reshape_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_43/strided_slice/stack_1
 reshape_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_43/strided_slice/stack_2Є
reshape_43/strided_sliceStridedSlicereshape_43/Shape:output:0'reshape_43/strided_slice/stack:output:0)reshape_43/strided_slice/stack_1:output:0)reshape_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_43/strided_slicez
reshape_43/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_43/Reshape/shape/1z
reshape_43/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_43/Reshape/shape/2з
reshape_43/Reshape/shapePack!reshape_43/strided_slice:output:0#reshape_43/Reshape/shape/1:output:0#reshape_43/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_43/Reshape/shapeЇ
reshape_43/ReshapeReshapedense_87/BiasAdd:output:0!reshape_43/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_43/Reshapeђ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_71_lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЧ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mulz
IdentityIdentityreshape_43/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp0^dense_87/bias/Regularizer/Square/ReadVariableOp$^lstm_71/lstm_cell_71/ReadVariableOp&^lstm_71/lstm_cell_71/ReadVariableOp_1&^lstm_71/lstm_cell_71/ReadVariableOp_2&^lstm_71/lstm_cell_71/ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*^lstm_71/lstm_cell_71/split/ReadVariableOp,^lstm_71/lstm_cell_71/split_1/ReadVariableOp^lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2J
#lstm_71/lstm_cell_71/ReadVariableOp#lstm_71/lstm_cell_71/ReadVariableOp2N
%lstm_71/lstm_cell_71/ReadVariableOp_1%lstm_71/lstm_cell_71/ReadVariableOp_12N
%lstm_71/lstm_cell_71/ReadVariableOp_2%lstm_71/lstm_cell_71/ReadVariableOp_22N
%lstm_71/lstm_cell_71/ReadVariableOp_3%lstm_71/lstm_cell_71/ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_71/lstm_cell_71/split/ReadVariableOp)lstm_71/lstm_cell_71/split/ReadVariableOp2Z
+lstm_71/lstm_cell_71/split_1/ReadVariableOp+lstm_71/lstm_cell_71/split_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ+
Г
J__inference_sequential_29_layer_call_and_return_conditional_losses_2303639

inputs"
lstm_71_2303565:	
lstm_71_2303567:	"
lstm_71_2303569:	 "
dense_86_2303584:  
dense_86_2303586: "
dense_87_2303606: 
dense_87_2303608:
identityЂ dense_86/StatefulPartitionedCallЂ dense_87/StatefulPartitionedCallЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂlstm_71/StatefulPartitionedCallЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЅ
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinputslstm_71_2303565lstm_71_2303567lstm_71_2303569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23035642!
lstm_71/StatefulPartitionedCallЙ
 dense_86/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_86_2303584dense_86_2303586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_23035832"
 dense_86/StatefulPartitionedCallК
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_2303606dense_87_2303608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_23036052"
 dense_87/StatefulPartitionedCall
reshape_43/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_43_layer_call_and_return_conditional_losses_23036242
reshape_43/PartitionedCallЯ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_71_2303565*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЏ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_2303608*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mul
IdentityIdentity#reshape_43/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall0^dense_87/bias/Regularizer/Square/ReadVariableOp ^lstm_71/StatefulPartitionedCall>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ+
Г
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304066

inputs"
lstm_71_2304035:	
lstm_71_2304037:	"
lstm_71_2304039:	 "
dense_86_2304042:  
dense_86_2304044: "
dense_87_2304047: 
dense_87_2304049:
identityЂ dense_86/StatefulPartitionedCallЂ dense_87/StatefulPartitionedCallЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂlstm_71/StatefulPartitionedCallЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЅ
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinputslstm_71_2304035lstm_71_2304037lstm_71_2304039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23040022!
lstm_71/StatefulPartitionedCallЙ
 dense_86/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_86_2304042dense_86_2304044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_23035832"
 dense_86/StatefulPartitionedCallК
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_2304047dense_87_2304049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_23036052"
 dense_87/StatefulPartitionedCall
reshape_43/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_43_layer_call_and_return_conditional_losses_23036242
reshape_43/PartitionedCallЯ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_71_2304035*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЏ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_2304049*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mul
IdentityIdentity#reshape_43/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall0^dense_87/bias/Regularizer/Square/ReadVariableOp ^lstm_71/StatefulPartitionedCall>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р
И
)__inference_lstm_71_layer_call_fn_2304881
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23030412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

і
E__inference_dense_86_layer_call_and_return_conditional_losses_2306023

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
И	
 
%__inference_signature_wrapper_2304209
input_30
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_23025312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
Ј
Ѕ	
while_body_2305013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Ѕ
while/lstm_cell_71/mulMulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЉ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Љ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Љ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ѓ

*__inference_dense_86_layer_call_fn_2306012

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_23035832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
э
Ј
E__inference_dense_87_layer_call_and_return_conditional_losses_2306054

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_87/bias/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddО
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_87/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЯR
ъ
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2302655

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3_
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulc
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1c
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2c
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6н
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
ф	
Ј
/__inference_sequential_29_layer_call_fn_2304228

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_29_layer_call_and_return_conditional_losses_23036392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ	
Њ
/__inference_sequential_29_layer_call_fn_2303656
input_30
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_29_layer_call_and_return_conditional_losses_23036392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
яЭ
Н
lstm_71_while_body_2304660,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0:	K
<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0:	G
4lstm_71_while_lstm_cell_71_readvariableop_resource_0:	 
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorK
8lstm_71_while_lstm_cell_71_split_readvariableop_resource:	I
:lstm_71_while_lstm_cell_71_split_1_readvariableop_resource:	E
2lstm_71_while_lstm_cell_71_readvariableop_resource:	 Ђ)lstm_71/while/lstm_cell_71/ReadVariableOpЂ+lstm_71/while/lstm_cell_71/ReadVariableOp_1Ђ+lstm_71/while/lstm_cell_71/ReadVariableOp_2Ђ+lstm_71/while/lstm_cell_71/ReadVariableOp_3Ђ/lstm_71/while/lstm_cell_71/split/ReadVariableOpЂ1lstm_71/while/lstm_cell_71/split_1/ReadVariableOpг
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_71/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_71/while/lstm_cell_71/ones_like/ShapeShapelstm_71_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_71/while/lstm_cell_71/ones_like/Shape
*lstm_71/while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_71/while/lstm_cell_71/ones_like/Const№
$lstm_71/while/lstm_cell_71/ones_likeFill3lstm_71/while/lstm_cell_71/ones_like/Shape:output:03lstm_71/while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/ones_like
(lstm_71/while/lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(lstm_71/while/lstm_cell_71/dropout/Constы
&lstm_71/while/lstm_cell_71/dropout/MulMul-lstm_71/while/lstm_cell_71/ones_like:output:01lstm_71/while/lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_71/while/lstm_cell_71/dropout/MulБ
(lstm_71/while/lstm_cell_71/dropout/ShapeShape-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_71/while/lstm_cell_71/dropout/ShapeЂ
?lstm_71/while/lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform1lstm_71/while/lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ПЏ2A
?lstm_71/while/lstm_cell_71/dropout/random_uniform/RandomUniformЋ
1lstm_71/while/lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1lstm_71/while/lstm_cell_71/dropout/GreaterEqual/yЊ
/lstm_71/while/lstm_cell_71/dropout/GreaterEqualGreaterEqualHlstm_71/while/lstm_cell_71/dropout/random_uniform/RandomUniform:output:0:lstm_71/while/lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_71/while/lstm_cell_71/dropout/GreaterEqualа
'lstm_71/while/lstm_cell_71/dropout/CastCast3lstm_71/while/lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_71/while/lstm_cell_71/dropout/Castц
(lstm_71/while/lstm_cell_71/dropout/Mul_1Mul*lstm_71/while/lstm_cell_71/dropout/Mul:z:0+lstm_71/while/lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_71/while/lstm_cell_71/dropout/Mul_1
*lstm_71/while/lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_71/while/lstm_cell_71/dropout_1/Constё
(lstm_71/while/lstm_cell_71/dropout_1/MulMul-lstm_71/while/lstm_cell_71/ones_like:output:03lstm_71/while/lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_71/while/lstm_cell_71/dropout_1/MulЕ
*lstm_71/while/lstm_cell_71/dropout_1/ShapeShape-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_71/while/lstm_cell_71/dropout_1/ShapeЇ
Alstm_71/while/lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_71/while/lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ь2C
Alstm_71/while/lstm_cell_71/dropout_1/random_uniform/RandomUniformЏ
3lstm_71/while/lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_71/while/lstm_cell_71/dropout_1/GreaterEqual/yВ
1lstm_71/while/lstm_cell_71/dropout_1/GreaterEqualGreaterEqualJlstm_71/while/lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:0<lstm_71/while/lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_71/while/lstm_cell_71/dropout_1/GreaterEqualж
)lstm_71/while/lstm_cell_71/dropout_1/CastCast5lstm_71/while/lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_71/while/lstm_cell_71/dropout_1/Castю
*lstm_71/while/lstm_cell_71/dropout_1/Mul_1Mul,lstm_71/while/lstm_cell_71/dropout_1/Mul:z:0-lstm_71/while/lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_71/while/lstm_cell_71/dropout_1/Mul_1
*lstm_71/while/lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_71/while/lstm_cell_71/dropout_2/Constё
(lstm_71/while/lstm_cell_71/dropout_2/MulMul-lstm_71/while/lstm_cell_71/ones_like:output:03lstm_71/while/lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_71/while/lstm_cell_71/dropout_2/MulЕ
*lstm_71/while/lstm_cell_71/dropout_2/ShapeShape-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_71/while/lstm_cell_71/dropout_2/ShapeЈ
Alstm_71/while/lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_71/while/lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2юРИ2C
Alstm_71/while/lstm_cell_71/dropout_2/random_uniform/RandomUniformЏ
3lstm_71/while/lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_71/while/lstm_cell_71/dropout_2/GreaterEqual/yВ
1lstm_71/while/lstm_cell_71/dropout_2/GreaterEqualGreaterEqualJlstm_71/while/lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:0<lstm_71/while/lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_71/while/lstm_cell_71/dropout_2/GreaterEqualж
)lstm_71/while/lstm_cell_71/dropout_2/CastCast5lstm_71/while/lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_71/while/lstm_cell_71/dropout_2/Castю
*lstm_71/while/lstm_cell_71/dropout_2/Mul_1Mul,lstm_71/while/lstm_cell_71/dropout_2/Mul:z:0-lstm_71/while/lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_71/while/lstm_cell_71/dropout_2/Mul_1
*lstm_71/while/lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_71/while/lstm_cell_71/dropout_3/Constё
(lstm_71/while/lstm_cell_71/dropout_3/MulMul-lstm_71/while/lstm_cell_71/ones_like:output:03lstm_71/while/lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_71/while/lstm_cell_71/dropout_3/MulЕ
*lstm_71/while/lstm_cell_71/dropout_3/ShapeShape-lstm_71/while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_71/while/lstm_cell_71/dropout_3/ShapeЇ
Alstm_71/while/lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_71/while/lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2нф2C
Alstm_71/while/lstm_cell_71/dropout_3/random_uniform/RandomUniformЏ
3lstm_71/while/lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_71/while/lstm_cell_71/dropout_3/GreaterEqual/yВ
1lstm_71/while/lstm_cell_71/dropout_3/GreaterEqualGreaterEqualJlstm_71/while/lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:0<lstm_71/while/lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_71/while/lstm_cell_71/dropout_3/GreaterEqualж
)lstm_71/while/lstm_cell_71/dropout_3/CastCast5lstm_71/while/lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_71/while/lstm_cell_71/dropout_3/Castю
*lstm_71/while/lstm_cell_71/dropout_3/Mul_1Mul,lstm_71/while/lstm_cell_71/dropout_3/Mul:z:0-lstm_71/while/lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_71/while/lstm_cell_71/dropout_3/Mul_1
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_71/while/lstm_cell_71/split/split_dimо
/lstm_71/while/lstm_cell_71/split/ReadVariableOpReadVariableOp:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_71/while/lstm_cell_71/split/ReadVariableOp
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:07lstm_71/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_71/while/lstm_cell_71/splitч
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_71/while/lstm_cell_71/MatMulы
#lstm_71/while/lstm_cell_71/MatMul_1MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_1ы
#lstm_71/while/lstm_cell_71/MatMul_2MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_2ы
#lstm_71/while/lstm_cell_71/MatMul_3MatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_3
,lstm_71/while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_71/while/lstm_cell_71/split_1/split_dimр
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp
"lstm_71/while/lstm_cell_71/split_1Split5lstm_71/while/lstm_cell_71/split_1/split_dim:output:09lstm_71/while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_71/while/lstm_cell_71/split_1п
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd+lstm_71/while/lstm_cell_71/MatMul:product:0+lstm_71/while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/while/lstm_cell_71/BiasAddх
$lstm_71/while/lstm_cell_71/BiasAdd_1BiasAdd-lstm_71/while/lstm_cell_71/MatMul_1:product:0+lstm_71/while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_1х
$lstm_71/while/lstm_cell_71/BiasAdd_2BiasAdd-lstm_71/while/lstm_cell_71/MatMul_2:product:0+lstm_71/while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_2х
$lstm_71/while/lstm_cell_71/BiasAdd_3BiasAdd-lstm_71/while/lstm_cell_71/MatMul_3:product:0+lstm_71/while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/BiasAdd_3Ф
lstm_71/while/lstm_cell_71/mulMullstm_71_while_placeholder_2,lstm_71/while/lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/while/lstm_cell_71/mulЪ
 lstm_71/while/lstm_cell_71/mul_1Mullstm_71_while_placeholder_2.lstm_71/while/lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_1Ъ
 lstm_71/while/lstm_cell_71/mul_2Mullstm_71_while_placeholder_2.lstm_71/while/lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_2Ъ
 lstm_71/while/lstm_cell_71/mul_3Mullstm_71_while_placeholder_2.lstm_71/while/lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_3Ь
)lstm_71/while/lstm_cell_71/ReadVariableOpReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_71/while/lstm_cell_71/ReadVariableOpБ
.lstm_71/while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_71/while/lstm_cell_71/strided_slice/stackЕ
0lstm_71/while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_71/while/lstm_cell_71/strided_slice/stack_1Е
0lstm_71/while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_71/while/lstm_cell_71/strided_slice/stack_2
(lstm_71/while/lstm_cell_71/strided_sliceStridedSlice1lstm_71/while/lstm_cell_71/ReadVariableOp:value:07lstm_71/while/lstm_cell_71/strided_slice/stack:output:09lstm_71/while/lstm_cell_71/strided_slice/stack_1:output:09lstm_71/while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_71/while/lstm_cell_71/strided_sliceн
#lstm_71/while/lstm_cell_71/MatMul_4MatMul"lstm_71/while/lstm_cell_71/mul:z:01lstm_71/while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_4з
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/BiasAdd:output:0-lstm_71/while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_71/while/lstm_cell_71/addЉ
"lstm_71/while/lstm_cell_71/SigmoidSigmoid"lstm_71/while/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_71/while/lstm_cell_71/Sigmoidа
+lstm_71/while/lstm_cell_71/ReadVariableOp_1ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_1Е
0lstm_71/while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_71/while/lstm_cell_71/strided_slice_1/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_1/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_1StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_1:value:09lstm_71/while/lstm_cell_71/strided_slice_1/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_1/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_1с
#lstm_71/while/lstm_cell_71/MatMul_5MatMul$lstm_71/while/lstm_cell_71/mul_1:z:03lstm_71/while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_5н
 lstm_71/while/lstm_cell_71/add_1AddV2-lstm_71/while/lstm_cell_71/BiasAdd_1:output:0-lstm_71/while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_1Џ
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/Sigmoid_1Ф
 lstm_71/while/lstm_cell_71/mul_4Mul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_4а
+lstm_71/while/lstm_cell_71/ReadVariableOp_2ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_2Е
0lstm_71/while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_71/while/lstm_cell_71/strided_slice_2/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_2/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_2StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_2:value:09lstm_71/while/lstm_cell_71/strided_slice_2/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_2/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_2с
#lstm_71/while/lstm_cell_71/MatMul_6MatMul$lstm_71/while/lstm_cell_71/mul_2:z:03lstm_71/while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_6н
 lstm_71/while/lstm_cell_71/add_2AddV2-lstm_71/while/lstm_cell_71/BiasAdd_2:output:0-lstm_71/while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_2Ђ
lstm_71/while/lstm_cell_71/ReluRelu$lstm_71/while/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_71/while/lstm_cell_71/Reluд
 lstm_71/while/lstm_cell_71/mul_5Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_5Ы
 lstm_71/while/lstm_cell_71/add_3AddV2$lstm_71/while/lstm_cell_71/mul_4:z:0$lstm_71/while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_3а
+lstm_71/while/lstm_cell_71/ReadVariableOp_3ReadVariableOp4lstm_71_while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_71/while/lstm_cell_71/ReadVariableOp_3Е
0lstm_71/while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_71/while/lstm_cell_71/strided_slice_3/stackЙ
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_1Й
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_71/while/lstm_cell_71/strided_slice_3/stack_2Њ
*lstm_71/while/lstm_cell_71/strided_slice_3StridedSlice3lstm_71/while/lstm_cell_71/ReadVariableOp_3:value:09lstm_71/while/lstm_cell_71/strided_slice_3/stack:output:0;lstm_71/while/lstm_cell_71/strided_slice_3/stack_1:output:0;lstm_71/while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_71/while/lstm_cell_71/strided_slice_3с
#lstm_71/while/lstm_cell_71/MatMul_7MatMul$lstm_71/while/lstm_cell_71/mul_3:z:03lstm_71/while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_71/while/lstm_cell_71/MatMul_7н
 lstm_71/while/lstm_cell_71/add_4AddV2-lstm_71/while/lstm_cell_71/BiasAdd_3:output:0-lstm_71/while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/add_4Џ
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid$lstm_71/while/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_71/while/lstm_cell_71/Sigmoid_2І
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_71/while/lstm_cell_71/Relu_1и
 lstm_71/while/lstm_cell_71/mul_6Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_71/while/lstm_cell_71/mul_6
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1lstm_71_while_placeholder$lstm_71/while/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_71/while/TensorArrayV2Write/TensorListSetIteml
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add/y
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/addp
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add_1/y
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/add_1
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/IdentityІ
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_1
lstm_71/while/Identity_2Identitylstm_71/while/add:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_2К
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_3­
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_6:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/while/Identity_4­
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_3:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_71/while/Identity_5
lstm_71/while/NoOpNoOp*^lstm_71/while/lstm_cell_71/ReadVariableOp,^lstm_71/while/lstm_cell_71/ReadVariableOp_1,^lstm_71/while/lstm_cell_71/ReadVariableOp_2,^lstm_71/while/lstm_cell_71/ReadVariableOp_30^lstm_71/while/lstm_cell_71/split/ReadVariableOp2^lstm_71/while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_71/while/NoOp"9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"j
2lstm_71_while_lstm_cell_71_readvariableop_resource4lstm_71_while_lstm_cell_71_readvariableop_resource_0"z
:lstm_71_while_lstm_cell_71_split_1_readvariableop_resource<lstm_71_while_lstm_cell_71_split_1_readvariableop_resource_0"v
8lstm_71_while_lstm_cell_71_split_readvariableop_resource:lstm_71_while_lstm_cell_71_split_readvariableop_resource_0"Ш
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_71/while/lstm_cell_71/ReadVariableOp)lstm_71/while/lstm_cell_71/ReadVariableOp2Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_1+lstm_71/while/lstm_cell_71/ReadVariableOp_12Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_2+lstm_71/while/lstm_cell_71/ReadVariableOp_22Z
+lstm_71/while/lstm_cell_71/ReadVariableOp_3+lstm_71/while/lstm_cell_71/ReadVariableOp_32b
/lstm_71/while/lstm_cell_71/split/ReadVariableOp/lstm_71/while/lstm_cell_71/split/ReadVariableOp2f
1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp1lstm_71/while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
э
Ј
E__inference_dense_87_layer_call_and_return_conditional_losses_2303605

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_87/bias/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddО
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_87/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ј
Ж
)__inference_lstm_71_layer_call_fn_2304903

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23040022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Єv
ъ
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2302888

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeа
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2з2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeж
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ш52(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeж
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2тЉf2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeж
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2БЧW2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3^
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
muld
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1d
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6н
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
Ы

ш
lstm_71_while_cond_2304659,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1E
Alstm_71_while_lstm_71_while_cond_2304659___redundant_placeholder0E
Alstm_71_while_lstm_71_while_cond_2304659___redundant_placeholder1E
Alstm_71_while_lstm_71_while_cond_2304659___redundant_placeholder2E
Alstm_71_while_lstm_71_while_cond_2304659___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2
lstm_71/while/Lessu
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_71/while/Identity"9
lstm_71_while_identitylstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ќВ
Ѕ	
while_body_2305838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
 while/lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_71/dropout/ConstЫ
while/lstm_cell_71/dropout/MulMul%while/lstm_cell_71/ones_like:output:0)while/lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_71/dropout/Mul
 while/lstm_cell_71/dropout/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_71/dropout/Shape
7while/lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2еВђ29
7while/lstm_cell_71/dropout/random_uniform/RandomUniform
)while/lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_71/dropout/GreaterEqual/y
'while/lstm_cell_71/dropout/GreaterEqualGreaterEqual@while/lstm_cell_71/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_71/dropout/GreaterEqualИ
while/lstm_cell_71/dropout/CastCast+while/lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_71/dropout/CastЦ
 while/lstm_cell_71/dropout/Mul_1Mul"while/lstm_cell_71/dropout/Mul:z:0#while/lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout/Mul_1
"while/lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_1/Constб
 while/lstm_cell_71/dropout_1/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_1/Mul
"while/lstm_cell_71/dropout_1/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_1/Shape
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЮОн2;
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_1/GreaterEqual/y
)while/lstm_cell_71/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_1/GreaterEqualО
!while/lstm_cell_71/dropout_1/CastCast-while/lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_1/CastЮ
"while/lstm_cell_71/dropout_1/Mul_1Mul$while/lstm_cell_71/dropout_1/Mul:z:0%while/lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_1/Mul_1
"while/lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_2/Constб
 while/lstm_cell_71/dropout_2/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_2/Mul
"while/lstm_cell_71/dropout_2/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_2/Shape
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ѕл2;
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_2/GreaterEqual/y
)while/lstm_cell_71/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_2/GreaterEqualО
!while/lstm_cell_71/dropout_2/CastCast-while/lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_2/CastЮ
"while/lstm_cell_71/dropout_2/Mul_1Mul$while/lstm_cell_71/dropout_2/Mul:z:0%while/lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_2/Mul_1
"while/lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_3/Constб
 while/lstm_cell_71/dropout_3/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_3/Mul
"while/lstm_cell_71/dropout_3/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_3/Shape
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2јЧ2;
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_3/GreaterEqual/y
)while/lstm_cell_71/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_3/GreaterEqualО
!while/lstm_cell_71/dropout_3/CastCast-while/lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_3/CastЮ
"while/lstm_cell_71/dropout_3/Mul_1Mul$while/lstm_cell_71/dropout_3/Mul:z:0%while/lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_3/Mul_1
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Є
while/lstm_cell_71/mulMulwhile_placeholder_2$while/lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЊ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2&while/lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Њ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2&while/lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Њ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2&while/lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
сЁ
Ј
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305696

inputs=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2305563*
condR
while_cond_2305562*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_2303430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2303430___redundant_placeholder05
1while_while_cond_2303430___redundant_placeholder15
1while_while_cond_2303430___redundant_placeholder25
1while_while_cond_2303430___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ћВ
Ѕ	
while_body_2303837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
 while/lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_71/dropout/ConstЫ
while/lstm_cell_71/dropout/MulMul%while/lstm_cell_71/ones_like:output:0)while/lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_71/dropout/Mul
 while/lstm_cell_71/dropout/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_71/dropout/Shape
7while/lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2иц29
7while/lstm_cell_71/dropout/random_uniform/RandomUniform
)while/lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_71/dropout/GreaterEqual/y
'while/lstm_cell_71/dropout/GreaterEqualGreaterEqual@while/lstm_cell_71/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_71/dropout/GreaterEqualИ
while/lstm_cell_71/dropout/CastCast+while/lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_71/dropout/CastЦ
 while/lstm_cell_71/dropout/Mul_1Mul"while/lstm_cell_71/dropout/Mul:z:0#while/lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout/Mul_1
"while/lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_1/Constб
 while/lstm_cell_71/dropout_1/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_1/Mul
"while/lstm_cell_71/dropout_1/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_1/Shape
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2оиК2;
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_1/GreaterEqual/y
)while/lstm_cell_71/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_1/GreaterEqualО
!while/lstm_cell_71/dropout_1/CastCast-while/lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_1/CastЮ
"while/lstm_cell_71/dropout_1/Mul_1Mul$while/lstm_cell_71/dropout_1/Mul:z:0%while/lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_1/Mul_1
"while/lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_2/Constб
 while/lstm_cell_71/dropout_2/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_2/Mul
"while/lstm_cell_71/dropout_2/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_2/Shape
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2МыМ2;
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_2/GreaterEqual/y
)while/lstm_cell_71/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_2/GreaterEqualО
!while/lstm_cell_71/dropout_2/CastCast-while/lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_2/CastЮ
"while/lstm_cell_71/dropout_2/Mul_1Mul$while/lstm_cell_71/dropout_2/Mul:z:0%while/lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_2/Mul_1
"while/lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_3/Constб
 while/lstm_cell_71/dropout_3/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_3/Mul
"while/lstm_cell_71/dropout_3/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_3/Shape
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2зе2;
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_3/GreaterEqual/y
)while/lstm_cell_71/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_3/GreaterEqualО
!while/lstm_cell_71/dropout_3/CastCast-while/lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_3/CastЮ
"while/lstm_cell_71/dropout_3/Mul_1Mul$while/lstm_cell_71/dropout_3/Mul:z:0%while/lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_3/Mul_1
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Є
while/lstm_cell_71/mulMulwhile_placeholder_2$while/lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЊ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2&while/lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Њ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2&while/lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Њ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2&while/lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 

і
E__inference_dense_86_layer_call_and_return_conditional_losses_2303583

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ы

ш
lstm_71_while_cond_2304356,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1E
Alstm_71_while_lstm_71_while_cond_2304356___redundant_placeholder0E
Alstm_71_while_lstm_71_while_cond_2304356___redundant_placeholder1E
Alstm_71_while_lstm_71_while_cond_2304356___redundant_placeholder2E
Alstm_71_while_lstm_71_while_cond_2304356___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2
lstm_71/while/Lessu
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_71/while/Identity"9
lstm_71_while_identitylstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ф	
Ј
/__inference_sequential_29_layer_call_fn_2304247

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_29_layer_call_and_return_conditional_losses_23040662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
G__inference_reshape_43_layer_call_and_return_conditional_losses_2303624

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
R
Щ
D__inference_lstm_71_layer_call_and_return_conditional_losses_2302744

inputs'
lstm_cell_71_2302656:	#
lstm_cell_71_2302658:	'
lstm_cell_71_2302660:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_71/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ё
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_2302656lstm_cell_71_2302658lstm_cell_71_2302660*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23026552&
$lstm_cell_71/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_2302656lstm_cell_71_2302658lstm_cell_71_2302660*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2302669*
condR
while_cond_2302668*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeд
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_71_2302656*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_71/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
Њ
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305146
inputs_0=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2305013*
condR
while_cond_2305012*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
н
Ы
__inference_loss_fn_1_2306328Y
Flstm_71_lstm_cell_71_kernel_regularizer_square_readvariableop_resource:	
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_71_lstm_cell_71_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muly
IdentityIdentity/lstm_71/lstm_cell_71/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp
ѓ

*__inference_dense_87_layer_call_fn_2306038

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_23036052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
к
Ш
while_cond_2302965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2302965___redundant_placeholder05
1while_while_cond_2302965___redundant_placeholder15
1while_while_cond_2302965___redundant_placeholder25
1while_while_cond_2302965___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
к
Ш
while_cond_2305837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2305837___redundant_placeholder05
1while_while_cond_2305837___redundant_placeholder15
1while_while_cond_2305837___redundant_placeholder25
1while_while_cond_2305837___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
мЯ
Ј
D__inference_lstm_71_layer_call_and_return_conditional_losses_2304002

inputs=
*lstm_cell_71_split_readvariableop_resource:	;
,lstm_cell_71_split_1_readvariableop_resource:	7
$lstm_cell_71_readvariableop_resource:	 
identityЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_71/ReadVariableOpЂlstm_cell_71/ReadVariableOp_1Ђlstm_cell_71/ReadVariableOp_2Ђlstm_cell_71/ReadVariableOp_3Ђ!lstm_cell_71/split/ReadVariableOpЂ#lstm_cell_71/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2z
lstm_cell_71/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_71/ones_like/Shape
lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_71/ones_like/ConstИ
lstm_cell_71/ones_likeFill%lstm_cell_71/ones_like/Shape:output:0%lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/ones_like}
lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout/ConstГ
lstm_cell_71/dropout/MulMullstm_cell_71/ones_like:output:0#lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul
lstm_cell_71/dropout/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout/Shapeј
1lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЃЈя23
1lstm_cell_71/dropout/random_uniform/RandomUniform
#lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_71/dropout/GreaterEqual/yђ
!lstm_cell_71/dropout/GreaterEqualGreaterEqual:lstm_cell_71/dropout/random_uniform/RandomUniform:output:0,lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_71/dropout/GreaterEqualІ
lstm_cell_71/dropout/CastCast%lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/CastЎ
lstm_cell_71/dropout/Mul_1Mullstm_cell_71/dropout/Mul:z:0lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout/Mul_1
lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_1/ConstЙ
lstm_cell_71/dropout_1/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul
lstm_cell_71/dropout_1/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_1/Shapeў
3lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2њЖж25
3lstm_cell_71/dropout_1/random_uniform/RandomUniform
%lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_1/GreaterEqual/yњ
#lstm_cell_71/dropout_1/GreaterEqualGreaterEqual<lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_1/GreaterEqualЌ
lstm_cell_71/dropout_1/CastCast'lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/CastЖ
lstm_cell_71/dropout_1/Mul_1Mullstm_cell_71/dropout_1/Mul:z:0lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_1/Mul_1
lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_2/ConstЙ
lstm_cell_71/dropout_2/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul
lstm_cell_71/dropout_2/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_2/Shapeў
3lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ћИЋ25
3lstm_cell_71/dropout_2/random_uniform/RandomUniform
%lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_2/GreaterEqual/yњ
#lstm_cell_71/dropout_2/GreaterEqualGreaterEqual<lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_2/GreaterEqualЌ
lstm_cell_71/dropout_2/CastCast'lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/CastЖ
lstm_cell_71/dropout_2/Mul_1Mullstm_cell_71/dropout_2/Mul:z:0lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_2/Mul_1
lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_71/dropout_3/ConstЙ
lstm_cell_71/dropout_3/MulMullstm_cell_71/ones_like:output:0%lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul
lstm_cell_71/dropout_3/ShapeShapelstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_71/dropout_3/Shapeў
3lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Д25
3lstm_cell_71/dropout_3/random_uniform/RandomUniform
%lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_71/dropout_3/GreaterEqual/yњ
#lstm_cell_71/dropout_3/GreaterEqualGreaterEqual<lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_71/dropout_3/GreaterEqualЌ
lstm_cell_71/dropout_3/CastCast'lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/CastЖ
lstm_cell_71/dropout_3/Mul_1Mullstm_cell_71/dropout_3/Mul:z:0lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/dropout_3/Mul_1~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimВ
!lstm_cell_71/split/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_71/split/ReadVariableOpл
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0)lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_71/split
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMulЁ
lstm_cell_71/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_1Ё
lstm_cell_71/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_2Ё
lstm_cell_71/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_3
lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_71/split_1/split_dimД
#lstm_cell_71/split_1/ReadVariableOpReadVariableOp,lstm_cell_71_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/split_1/ReadVariableOpг
lstm_cell_71/split_1Split'lstm_cell_71/split_1/split_dim:output:0+lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_71/split_1Ї
lstm_cell_71/BiasAddBiasAddlstm_cell_71/MatMul:product:0lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd­
lstm_cell_71/BiasAdd_1BiasAddlstm_cell_71/MatMul_1:product:0lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_1­
lstm_cell_71/BiasAdd_2BiasAddlstm_cell_71/MatMul_2:product:0lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_2­
lstm_cell_71/BiasAdd_3BiasAddlstm_cell_71/MatMul_3:product:0lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/BiasAdd_3
lstm_cell_71/mulMulzeros:output:0lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul
lstm_cell_71/mul_1Mulzeros:output:0 lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_1
lstm_cell_71/mul_2Mulzeros:output:0 lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_2
lstm_cell_71/mul_3Mulzeros:output:0 lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_3 
lstm_cell_71/ReadVariableOpReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp
 lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_71/strided_slice/stack
"lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice/stack_1
"lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_71/strided_slice/stack_2Ъ
lstm_cell_71/strided_sliceStridedSlice#lstm_cell_71/ReadVariableOp:value:0)lstm_cell_71/strided_slice/stack:output:0+lstm_cell_71/strided_slice/stack_1:output:0+lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_sliceЅ
lstm_cell_71/MatMul_4MatMullstm_cell_71/mul:z:0#lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_4
lstm_cell_71/addAddV2lstm_cell_71/BiasAdd:output:0lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add
lstm_cell_71/SigmoidSigmoidlstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/SigmoidЄ
lstm_cell_71/ReadVariableOp_1ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_1
"lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_71/strided_slice_1/stack
$lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_71/strided_slice_1/stack_1
$lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_1/stack_2ж
lstm_cell_71/strided_slice_1StridedSlice%lstm_cell_71/ReadVariableOp_1:value:0+lstm_cell_71/strided_slice_1/stack:output:0-lstm_cell_71/strided_slice_1/stack_1:output:0-lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_1Љ
lstm_cell_71/MatMul_5MatMullstm_cell_71/mul_1:z:0%lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_5Ѕ
lstm_cell_71/add_1AddV2lstm_cell_71/BiasAdd_1:output:0lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mul_4Mullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_4Є
lstm_cell_71/ReadVariableOp_2ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_2
"lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_71/strided_slice_2/stack
$lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_71/strided_slice_2/stack_1
$lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_2/stack_2ж
lstm_cell_71/strided_slice_2StridedSlice%lstm_cell_71/ReadVariableOp_2:value:0+lstm_cell_71/strided_slice_2/stack:output:0-lstm_cell_71/strided_slice_2/stack_1:output:0-lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_2Љ
lstm_cell_71/MatMul_6MatMullstm_cell_71/mul_2:z:0%lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_6Ѕ
lstm_cell_71/add_2AddV2lstm_cell_71/BiasAdd_2:output:0lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_2x
lstm_cell_71/ReluRelulstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu
lstm_cell_71/mul_5Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_5
lstm_cell_71/add_3AddV2lstm_cell_71/mul_4:z:0lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_3Є
lstm_cell_71/ReadVariableOp_3ReadVariableOp$lstm_cell_71_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_71/ReadVariableOp_3
"lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_71/strided_slice_3/stack
$lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_71/strided_slice_3/stack_1
$lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_71/strided_slice_3/stack_2ж
lstm_cell_71/strided_slice_3StridedSlice%lstm_cell_71/ReadVariableOp_3:value:0+lstm_cell_71/strided_slice_3/stack:output:0-lstm_cell_71/strided_slice_3/stack_1:output:0-lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_71/strided_slice_3Љ
lstm_cell_71/MatMul_7MatMullstm_cell_71/mul_3:z:0%lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/MatMul_7Ѕ
lstm_cell_71/add_4AddV2lstm_cell_71/BiasAdd_3:output:0lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/add_4
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_6Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_71/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_71_split_readvariableop_resource,lstm_cell_71_split_1_readvariableop_resource$lstm_cell_71_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2303837*
condR
while_cond_2303836*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_71_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_71/ReadVariableOp^lstm_cell_71/ReadVariableOp_1^lstm_cell_71/ReadVariableOp_2^lstm_cell_71/ReadVariableOp_3"^lstm_cell_71/split/ReadVariableOp$^lstm_cell_71/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_71/ReadVariableOplstm_cell_71/ReadVariableOp2>
lstm_cell_71/ReadVariableOp_1lstm_cell_71/ReadVariableOp_12>
lstm_cell_71/ReadVariableOp_2lstm_cell_71/ReadVariableOp_22>
lstm_cell_71/ReadVariableOp_3lstm_cell_71/ReadVariableOp_32F
!lstm_cell_71/split/ReadVariableOp!lstm_cell_71/split/ReadVariableOp2J
#lstm_cell_71/split_1/ReadVariableOp#lstm_cell_71/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№

(sequential_29_lstm_71_while_cond_2302381H
Dsequential_29_lstm_71_while_sequential_29_lstm_71_while_loop_counterN
Jsequential_29_lstm_71_while_sequential_29_lstm_71_while_maximum_iterations+
'sequential_29_lstm_71_while_placeholder-
)sequential_29_lstm_71_while_placeholder_1-
)sequential_29_lstm_71_while_placeholder_2-
)sequential_29_lstm_71_while_placeholder_3J
Fsequential_29_lstm_71_while_less_sequential_29_lstm_71_strided_slice_1a
]sequential_29_lstm_71_while_sequential_29_lstm_71_while_cond_2302381___redundant_placeholder0a
]sequential_29_lstm_71_while_sequential_29_lstm_71_while_cond_2302381___redundant_placeholder1a
]sequential_29_lstm_71_while_sequential_29_lstm_71_while_cond_2302381___redundant_placeholder2a
]sequential_29_lstm_71_while_sequential_29_lstm_71_while_cond_2302381___redundant_placeholder3(
$sequential_29_lstm_71_while_identity
о
 sequential_29/lstm_71/while/LessLess'sequential_29_lstm_71_while_placeholderFsequential_29_lstm_71_while_less_sequential_29_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_29/lstm_71/while/Less
$sequential_29/lstm_71/while/IdentityIdentity$sequential_29/lstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_29/lstm_71/while/Identity"U
$sequential_29_lstm_71_while_identity-sequential_29/lstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ј+
Е
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304136
input_30"
lstm_71_2304105:	
lstm_71_2304107:	"
lstm_71_2304109:	 "
dense_86_2304112:  
dense_86_2304114: "
dense_87_2304117: 
dense_87_2304119:
identityЂ dense_86/StatefulPartitionedCallЂ dense_87/StatefulPartitionedCallЂ/dense_87/bias/Regularizer/Square/ReadVariableOpЂlstm_71/StatefulPartitionedCallЂ=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpЇ
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinput_30lstm_71_2304105lstm_71_2304107lstm_71_2304109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23035642!
lstm_71/StatefulPartitionedCallЙ
 dense_86/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_86_2304112dense_86_2304114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_23035832"
 dense_86/StatefulPartitionedCallК
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_2304117dense_87_2304119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_23036052"
 dense_87/StatefulPartitionedCall
reshape_43/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_43_layer_call_and_return_conditional_losses_23036242
reshape_43/PartitionedCallЯ
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_71_2304105*
_output_shapes
:	*
dtype02?
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOpл
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareSquareElstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_71/lstm_cell_71/kernel/Regularizer/SquareЏ
-lstm_71/lstm_cell_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_71/lstm_cell_71/kernel/Regularizer/Constю
+lstm_71/lstm_cell_71/kernel/Regularizer/SumSum2lstm_71/lstm_cell_71/kernel/Regularizer/Square:y:06lstm_71/lstm_cell_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/SumЃ
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_71/lstm_cell_71/kernel/Regularizer/mul/x№
+lstm_71/lstm_cell_71/kernel/Regularizer/mulMul6lstm_71/lstm_cell_71/kernel/Regularizer/mul/x:output:04lstm_71/lstm_cell_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_71/lstm_cell_71/kernel/Regularizer/mulЏ
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_2304119*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mul
IdentityIdentity#reshape_43/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall0^dense_87/bias/Regularizer/Square/ReadVariableOp ^lstm_71/StatefulPartitionedCall>^lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall2~
=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp=lstm_71/lstm_cell_71/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_30
к
Ш
while_cond_2302668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2302668___redundant_placeholder05
1while_while_cond_2302668___redundant_placeholder15
1while_while_cond_2302668___redundant_placeholder25
1while_while_cond_2302668___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ЭB
у
 __inference__traced_save_2306435
file_prefix.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableopD
@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop8
4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_86_kernel_m_read_readvariableop3
/savev2_adam_dense_86_bias_m_read_readvariableop5
1savev2_adam_dense_87_kernel_m_read_readvariableop3
/savev2_adam_dense_87_bias_m_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop5
1savev2_adam_dense_86_kernel_v_read_readvariableop3
/savev2_adam_dense_86_bias_v_read_readvariableop5
1savev2_adam_dense_87_kernel_v_read_readvariableop3
/savev2_adam_dense_87_bias_v_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameа
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
valueиBеB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesм
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableop@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_86_kernel_m_read_readvariableop/savev2_adam_dense_86_bias_m_read_readvariableop1savev2_adam_dense_87_kernel_m_read_readvariableop/savev2_adam_dense_87_bias_m_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop1savev2_adam_dense_86_kernel_v_read_readvariableop/savev2_adam_dense_86_bias_v_read_readvariableop1savev2_adam_dense_87_kernel_v_read_readvariableop/savev2_adam_dense_87_bias_v_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*о
_input_shapesЬ
Щ: :  : : :: : : : : :	:	 :: : :  : : ::	:	 ::  : : ::	:	 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: 
к
Ш
while_cond_2305012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2305012___redundant_placeholder05
1while_while_cond_2305012___redundant_placeholder15
1while_while_cond_2305012___redundant_placeholder25
1while_while_cond_2305012___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ќВ
Ѕ	
while_body_2305288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
 while/lstm_cell_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_71/dropout/ConstЫ
while/lstm_cell_71/dropout/MulMul%while/lstm_cell_71/ones_like:output:0)while/lstm_cell_71/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_71/dropout/Mul
 while/lstm_cell_71/dropout/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_71/dropout/Shape
7while/lstm_cell_71/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_71/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЕЪЋ29
7while/lstm_cell_71/dropout/random_uniform/RandomUniform
)while/lstm_cell_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_71/dropout/GreaterEqual/y
'while/lstm_cell_71/dropout/GreaterEqualGreaterEqual@while/lstm_cell_71/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_71/dropout/GreaterEqualИ
while/lstm_cell_71/dropout/CastCast+while/lstm_cell_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_71/dropout/CastЦ
 while/lstm_cell_71/dropout/Mul_1Mul"while/lstm_cell_71/dropout/Mul:z:0#while/lstm_cell_71/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout/Mul_1
"while/lstm_cell_71/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_1/Constб
 while/lstm_cell_71/dropout_1/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_1/Mul
"while/lstm_cell_71/dropout_1/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_1/Shape
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2јБЋ2;
9while/lstm_cell_71/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_1/GreaterEqual/y
)while/lstm_cell_71/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_1/GreaterEqualО
!while/lstm_cell_71/dropout_1/CastCast-while/lstm_cell_71/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_1/CastЮ
"while/lstm_cell_71/dropout_1/Mul_1Mul$while/lstm_cell_71/dropout_1/Mul:z:0%while/lstm_cell_71/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_1/Mul_1
"while/lstm_cell_71/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_2/Constб
 while/lstm_cell_71/dropout_2/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_2/Mul
"while/lstm_cell_71/dropout_2/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_2/Shape
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2нГЃ2;
9while/lstm_cell_71/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_2/GreaterEqual/y
)while/lstm_cell_71/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_2/GreaterEqualО
!while/lstm_cell_71/dropout_2/CastCast-while/lstm_cell_71/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_2/CastЮ
"while/lstm_cell_71/dropout_2/Mul_1Mul$while/lstm_cell_71/dropout_2/Mul:z:0%while/lstm_cell_71/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_2/Mul_1
"while/lstm_cell_71/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_71/dropout_3/Constб
 while/lstm_cell_71/dropout_3/MulMul%while/lstm_cell_71/ones_like:output:0+while/lstm_cell_71/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_71/dropout_3/Mul
"while/lstm_cell_71/dropout_3/ShapeShape%while/lstm_cell_71/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_71/dropout_3/Shape
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_71/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2еЪТ2;
9while/lstm_cell_71/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_71/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_71/dropout_3/GreaterEqual/y
)while/lstm_cell_71/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_71/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_71/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_71/dropout_3/GreaterEqualО
!while/lstm_cell_71/dropout_3/CastCast-while/lstm_cell_71/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_71/dropout_3/CastЮ
"while/lstm_cell_71/dropout_3/Mul_1Mul$while/lstm_cell_71/dropout_3/Mul:z:0%while/lstm_cell_71/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_71/dropout_3/Mul_1
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Є
while/lstm_cell_71/mulMulwhile_placeholder_2$while/lstm_cell_71/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЊ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2&while/lstm_cell_71/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Њ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2&while/lstm_cell_71/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Њ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2&while/lstm_cell_71/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Ъ
H
,__inference_reshape_43_layer_call_fn_2306059

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_43_layer_call_and_return_conditional_losses_23036242
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р
И
)__inference_lstm_71_layer_call_fn_2304870
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_23027442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ј
Ѕ	
while_body_2305563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_71_split_readvariableop_resource_0:	C
4while_lstm_cell_71_split_1_readvariableop_resource_0:	?
,while_lstm_cell_71_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_71_split_readvariableop_resource:	A
2while_lstm_cell_71_split_1_readvariableop_resource:	=
*while_lstm_cell_71_readvariableop_resource:	 Ђ!while/lstm_cell_71/ReadVariableOpЂ#while/lstm_cell_71/ReadVariableOp_1Ђ#while/lstm_cell_71/ReadVariableOp_2Ђ#while/lstm_cell_71/ReadVariableOp_3Ђ'while/lstm_cell_71/split/ReadVariableOpЂ)while/lstm_cell_71/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
"while/lstm_cell_71/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_71/ones_like/Shape
"while/lstm_cell_71/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_71/ones_like/Constа
while/lstm_cell_71/ones_likeFill+while/lstm_cell_71/ones_like/Shape:output:0+while/lstm_cell_71/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ones_like
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dimЦ
'while/lstm_cell_71/split/ReadVariableOpReadVariableOp2while_lstm_cell_71_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_71/split/ReadVariableOpѓ
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0/while/lstm_cell_71/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_71/splitЧ
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMulЫ
while/lstm_cell_71/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_1Ы
while/lstm_cell_71/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_2Ы
while/lstm_cell_71/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_3
$while/lstm_cell_71/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_71/split_1/split_dimШ
)while/lstm_cell_71/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_71_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/split_1/ReadVariableOpы
while/lstm_cell_71/split_1Split-while/lstm_cell_71/split_1/split_dim:output:01while/lstm_cell_71/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_71/split_1П
while/lstm_cell_71/BiasAddBiasAdd#while/lstm_cell_71/MatMul:product:0#while/lstm_cell_71/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAddХ
while/lstm_cell_71/BiasAdd_1BiasAdd%while/lstm_cell_71/MatMul_1:product:0#while/lstm_cell_71/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_1Х
while/lstm_cell_71/BiasAdd_2BiasAdd%while/lstm_cell_71/MatMul_2:product:0#while/lstm_cell_71/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_2Х
while/lstm_cell_71/BiasAdd_3BiasAdd%while/lstm_cell_71/MatMul_3:product:0#while/lstm_cell_71/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/BiasAdd_3Ѕ
while/lstm_cell_71/mulMulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mulЉ
while/lstm_cell_71/mul_1Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_1Љ
while/lstm_cell_71/mul_2Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_2Љ
while/lstm_cell_71/mul_3Mulwhile_placeholder_2%while/lstm_cell_71/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_3Д
!while/lstm_cell_71/ReadVariableOpReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_71/ReadVariableOpЁ
&while/lstm_cell_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_71/strided_slice/stackЅ
(while/lstm_cell_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice/stack_1Ѕ
(while/lstm_cell_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_71/strided_slice/stack_2ю
 while/lstm_cell_71/strided_sliceStridedSlice)while/lstm_cell_71/ReadVariableOp:value:0/while/lstm_cell_71/strided_slice/stack:output:01while/lstm_cell_71/strided_slice/stack_1:output:01while/lstm_cell_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_71/strided_sliceН
while/lstm_cell_71/MatMul_4MatMulwhile/lstm_cell_71/mul:z:0)while/lstm_cell_71/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_4З
while/lstm_cell_71/addAddV2#while/lstm_cell_71/BiasAdd:output:0%while/lstm_cell_71/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add
while/lstm_cell_71/SigmoidSigmoidwhile/lstm_cell_71/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/SigmoidИ
#while/lstm_cell_71/ReadVariableOp_1ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_1Ѕ
(while/lstm_cell_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_71/strided_slice_1/stackЉ
*while/lstm_cell_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_71/strided_slice_1/stack_1Љ
*while/lstm_cell_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_1/stack_2њ
"while/lstm_cell_71/strided_slice_1StridedSlice+while/lstm_cell_71/ReadVariableOp_1:value:01while/lstm_cell_71/strided_slice_1/stack:output:03while/lstm_cell_71/strided_slice_1/stack_1:output:03while/lstm_cell_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_1С
while/lstm_cell_71/MatMul_5MatMulwhile/lstm_cell_71/mul_1:z:0+while/lstm_cell_71/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_5Н
while/lstm_cell_71/add_1AddV2%while/lstm_cell_71/BiasAdd_1:output:0%while/lstm_cell_71/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_1Sigmoidwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_1Є
while/lstm_cell_71/mul_4Mul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_4И
#while/lstm_cell_71/ReadVariableOp_2ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_2Ѕ
(while/lstm_cell_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_71/strided_slice_2/stackЉ
*while/lstm_cell_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_71/strided_slice_2/stack_1Љ
*while/lstm_cell_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_2/stack_2њ
"while/lstm_cell_71/strided_slice_2StridedSlice+while/lstm_cell_71/ReadVariableOp_2:value:01while/lstm_cell_71/strided_slice_2/stack:output:03while/lstm_cell_71/strided_slice_2/stack_1:output:03while/lstm_cell_71/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_2С
while/lstm_cell_71/MatMul_6MatMulwhile/lstm_cell_71/mul_2:z:0+while/lstm_cell_71/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_6Н
while/lstm_cell_71/add_2AddV2%while/lstm_cell_71/BiasAdd_2:output:0%while/lstm_cell_71/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_2
while/lstm_cell_71/ReluReluwhile/lstm_cell_71/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/ReluД
while/lstm_cell_71/mul_5Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_5Ћ
while/lstm_cell_71/add_3AddV2while/lstm_cell_71/mul_4:z:0while/lstm_cell_71/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_3И
#while/lstm_cell_71/ReadVariableOp_3ReadVariableOp,while_lstm_cell_71_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_71/ReadVariableOp_3Ѕ
(while/lstm_cell_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_71/strided_slice_3/stackЉ
*while/lstm_cell_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_71/strided_slice_3/stack_1Љ
*while/lstm_cell_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_71/strided_slice_3/stack_2њ
"while/lstm_cell_71/strided_slice_3StridedSlice+while/lstm_cell_71/ReadVariableOp_3:value:01while/lstm_cell_71/strided_slice_3/stack:output:03while/lstm_cell_71/strided_slice_3/stack_1:output:03while/lstm_cell_71/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_71/strided_slice_3С
while/lstm_cell_71/MatMul_7MatMulwhile/lstm_cell_71/mul_3:z:0+while/lstm_cell_71/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/MatMul_7Н
while/lstm_cell_71/add_4AddV2%while/lstm_cell_71/BiasAdd_3:output:0%while/lstm_cell_71/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/add_4
while/lstm_cell_71/Sigmoid_2Sigmoidwhile/lstm_cell_71/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/Relu_1И
while/lstm_cell_71/mul_6Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_71/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_71/ReadVariableOp$^while/lstm_cell_71/ReadVariableOp_1$^while/lstm_cell_71/ReadVariableOp_2$^while/lstm_cell_71/ReadVariableOp_3(^while/lstm_cell_71/split/ReadVariableOp*^while/lstm_cell_71/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_71_readvariableop_resource,while_lstm_cell_71_readvariableop_resource_0"j
2while_lstm_cell_71_split_1_readvariableop_resource4while_lstm_cell_71_split_1_readvariableop_resource_0"f
0while_lstm_cell_71_split_readvariableop_resource2while_lstm_cell_71_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_71/ReadVariableOp!while/lstm_cell_71/ReadVariableOp2J
#while/lstm_cell_71/ReadVariableOp_1#while/lstm_cell_71/ReadVariableOp_12J
#while/lstm_cell_71/ReadVariableOp_2#while/lstm_cell_71/ReadVariableOp_22J
#while/lstm_cell_71/ReadVariableOp_3#while/lstm_cell_71/ReadVariableOp_32R
'while/lstm_cell_71/split/ReadVariableOp'while/lstm_cell_71/split/ReadVariableOp2V
)while/lstm_cell_71/split_1/ReadVariableOp)while/lstm_cell_71/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
И
ї
.__inference_lstm_cell_71_layer_call_fn_2306106

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23026552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
ї
Њ
__inference_loss_fn_0_2306083F
8dense_87_bias_regularizer_square_readvariableop_resource:
identityЂ/dense_87/bias/Regularizer/Square/ReadVariableOpз
/dense_87/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_87_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_87/bias/Regularizer/Square/ReadVariableOpЌ
 dense_87/bias/Regularizer/SquareSquare7dense_87/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_87/bias/Regularizer/Square
dense_87/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_87/bias/Regularizer/ConstЖ
dense_87/bias/Regularizer/SumSum$dense_87/bias/Regularizer/Square:y:0(dense_87/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/Sum
dense_87/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_87/bias/Regularizer/mul/xИ
dense_87/bias/Regularizer/mulMul(dense_87/bias/Regularizer/mul/x:output:0&dense_87/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_87/bias/Regularizer/mulk
IdentityIdentity!dense_87/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense_87/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_87/bias/Regularizer/Square/ReadVariableOp/dense_87/bias/Regularizer/Square/ReadVariableOp
И
ї
.__inference_lstm_cell_71_layer_call_fn_2306123

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_23028882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*З
serving_defaultЃ
A
input_305
serving_default_input_30:0џџџџџџџџџB

reshape_434
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:н
ш
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
`_default_save_signature
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_sequential
У
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Л

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
trainable_variables
	variables
regularization_losses
 	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
б
!iter

"beta_1

#beta_2
	$decay
%learning_ratemRmSmTmU&mV'mW(mXvYvZv[v\&v]'v^(v_"
	optimizer
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
Ъ

)layers
trainable_variables
	variables
*metrics
+layer_metrics
,layer_regularization_losses
-non_trainable_variables
regularization_losses
a__call__
`_default_save_signature
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
,
lserving_default"
signature_map
с
.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
Й

3layers
trainable_variables
	variables
4metrics
5layer_metrics
6layer_regularization_losses
7non_trainable_variables

8states
regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_86/kernel
: 2dense_86/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

9layers
trainable_variables
	variables
:metrics
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_87/kernel
:2dense_87/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
­

>layers
trainable_variables
	variables
?metrics
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Clayers
trainable_variables
	variables
Dmetrics
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables
regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	2lstm_71/lstm_cell_71/kernel
8:6	 2%lstm_71/lstm_cell_71/recurrent_kernel
(:&2lstm_71/lstm_cell_71/bias
<
0
1
2
3"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
­

Ilayers
/trainable_variables
0	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
Mnon_trainable_variables
1regularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Ntotal
	Ocount
P	variables
Q	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
N0
O1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
&:$  2Adam/dense_86/kernel/m
 : 2Adam/dense_86/bias/m
&:$ 2Adam/dense_87/kernel/m
 :2Adam/dense_87/bias/m
3:1	2"Adam/lstm_71/lstm_cell_71/kernel/m
=:;	 2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
-:+2 Adam/lstm_71/lstm_cell_71/bias/m
&:$  2Adam/dense_86/kernel/v
 : 2Adam/dense_86/bias/v
&:$ 2Adam/dense_87/kernel/v
 :2Adam/dense_87/bias/v
3:1	2"Adam/lstm_71/lstm_cell_71/kernel/v
=:;	 2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
-:+2 Adam/lstm_71/lstm_cell_71/bias/v
ЮBЫ
"__inference__wrapped_model_2302531input_30"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
/__inference_sequential_29_layer_call_fn_2303656
/__inference_sequential_29_layer_call_fn_2304228
/__inference_sequential_29_layer_call_fn_2304247
/__inference_sequential_29_layer_call_fn_2304102Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304518
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304853
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304136
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304170Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_lstm_71_layer_call_fn_2304870
)__inference_lstm_71_layer_call_fn_2304881
)__inference_lstm_71_layer_call_fn_2304892
)__inference_lstm_71_layer_call_fn_2304903е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ѓ2№
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305146
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305453
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305696
D__inference_lstm_71_layer_call_and_return_conditional_losses_2306003е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_86_layer_call_fn_2306012Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_86_layer_call_and_return_conditional_losses_2306023Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_87_layer_call_fn_2306038Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_87_layer_call_and_return_conditional_losses_2306054Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_reshape_43_layer_call_fn_2306059Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_reshape_43_layer_call_and_return_conditional_losses_2306072Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2Б
__inference_loss_fn_0_2306083
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ЭBЪ
%__inference_signature_wrapper_2304209input_30"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
.__inference_lstm_cell_71_layer_call_fn_2306106
.__inference_lstm_cell_71_layer_call_fn_2306123О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306204
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306317О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Д2Б
__inference_loss_fn_1_2306328
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ Ѓ
"__inference__wrapped_model_2302531}&('5Ђ2
+Ђ(
&#
input_30џџџџџџџџџ
Њ ";Њ8
6

reshape_43(%

reshape_43џџџџџџџџџЅ
E__inference_dense_86_layer_call_and_return_conditional_losses_2306023\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_86_layer_call_fn_2306012O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ѕ
E__inference_dense_87_layer_call_and_return_conditional_losses_2306054\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_87_layer_call_fn_2306038O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ<
__inference_loss_fn_0_2306083Ђ

Ђ 
Њ " <
__inference_loss_fn_1_2306328&Ђ

Ђ 
Њ " Х
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305146}&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 Х
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305453}&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 Е
D__inference_lstm_71_layer_call_and_return_conditional_losses_2305696m&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 Е
D__inference_lstm_71_layer_call_and_return_conditional_losses_2306003m&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 
)__inference_lstm_71_layer_call_fn_2304870p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
)__inference_lstm_71_layer_call_fn_2304881p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ 
)__inference_lstm_71_layer_call_fn_2304892`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
)__inference_lstm_71_layer_call_fn_2304903`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ Ы
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306204§&('Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ 
EB

0/1/0џџџџџџџџџ 

0/1/1џџџџџџџџџ 
 Ы
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_2306317§&('Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ 
EB

0/1/0џџџџџџџџџ 

0/1/1џџџџџџџџџ 
  
.__inference_lstm_cell_71_layer_call_fn_2306106э&('Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p 
Њ "cЂ`

0џџџџџџџџџ 
A>

1/0џџџџџџџџџ 

1/1џџџџџџџџџ  
.__inference_lstm_cell_71_layer_call_fn_2306123э&('Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p
Њ "cЂ`

0џџџџџџџџџ 
A>

1/0џџџџџџџџџ 

1/1џџџџџџџџџ Ї
G__inference_reshape_43_layer_call_and_return_conditional_losses_2306072\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
,__inference_reshape_43_layer_call_fn_2306059O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџС
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304136s&('=Ђ:
3Ђ0
&#
input_30џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 С
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304170s&('=Ђ:
3Ђ0
&#
input_30џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 П
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304518q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 П
J__inference_sequential_29_layer_call_and_return_conditional_losses_2304853q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 
/__inference_sequential_29_layer_call_fn_2303656f&('=Ђ:
3Ђ0
&#
input_30џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_29_layer_call_fn_2304102f&('=Ђ:
3Ђ0
&#
input_30џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_29_layer_call_fn_2304228d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_29_layer_call_fn_2304247d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџГ
%__inference_signature_wrapper_2304209&('AЂ>
Ђ 
7Њ4
2
input_30&#
input_30џџџџџџџџџ";Њ8
6

reshape_43(%

reshape_43џџџџџџџџџ