Ею&
╦Ь
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
л
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8щ╬%
z
dense_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_92/kernel
s
#dense_92/kernel/Read/ReadVariableOpReadVariableOpdense_92/kernel*
_output_shapes

:  *
dtype0
r
dense_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_92/bias
k
!dense_92/bias/Read/ReadVariableOpReadVariableOpdense_92/bias*
_output_shapes
: *
dtype0
z
dense_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_93/kernel
s
#dense_93/kernel/Read/ReadVariableOpReadVariableOpdense_93/kernel*
_output_shapes

: *
dtype0
r
dense_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_93/bias
k
!dense_93/bias/Read/ReadVariableOpReadVariableOpdense_93/bias*
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
У
lstm_76/lstm_cell_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_namelstm_76/lstm_cell_76/kernel
М
/lstm_76/lstm_cell_76/kernel/Read/ReadVariableOpReadVariableOplstm_76/lstm_cell_76/kernel*
_output_shapes
:	А*
dtype0
з
%lstm_76/lstm_cell_76/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*6
shared_name'%lstm_76/lstm_cell_76/recurrent_kernel
а
9lstm_76/lstm_cell_76/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_76/lstm_cell_76/recurrent_kernel*
_output_shapes
:	 А*
dtype0
Л
lstm_76/lstm_cell_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_76/lstm_cell_76/bias
Д
-lstm_76/lstm_cell_76/bias/Read/ReadVariableOpReadVariableOplstm_76/lstm_cell_76/bias*
_output_shapes	
:А*
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
И
Adam/dense_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_92/kernel/m
Б
*Adam/dense_92/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/m*
_output_shapes

:  *
dtype0
А
Adam/dense_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_92/bias/m
y
(Adam/dense_92/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/m*
_output_shapes
: *
dtype0
И
Adam/dense_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_93/kernel/m
Б
*Adam/dense_93/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_93/bias/m
y
(Adam/dense_93/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/m*
_output_shapes
:*
dtype0
б
"Adam/lstm_76/lstm_cell_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_76/lstm_cell_76/kernel/m
Ъ
6Adam/lstm_76/lstm_cell_76/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_76/lstm_cell_76/kernel/m*
_output_shapes
:	А*
dtype0
╡
,Adam/lstm_76/lstm_cell_76/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/lstm_76/lstm_cell_76/recurrent_kernel/m
о
@Adam/lstm_76/lstm_cell_76/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_76/lstm_cell_76/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Щ
 Adam/lstm_76/lstm_cell_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_76/lstm_cell_76/bias/m
Т
4Adam/lstm_76/lstm_cell_76/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_76/lstm_cell_76/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_92/kernel/v
Б
*Adam/dense_92/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/v*
_output_shapes

:  *
dtype0
А
Adam/dense_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_92/bias/v
y
(Adam/dense_92/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/v*
_output_shapes
: *
dtype0
И
Adam/dense_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_93/kernel/v
Б
*Adam/dense_93/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_93/bias/v
y
(Adam/dense_93/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/v*
_output_shapes
:*
dtype0
б
"Adam/lstm_76/lstm_cell_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_76/lstm_cell_76/kernel/v
Ъ
6Adam/lstm_76/lstm_cell_76/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_76/lstm_cell_76/kernel/v*
_output_shapes
:	А*
dtype0
╡
,Adam/lstm_76/lstm_cell_76/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/lstm_76/lstm_cell_76/recurrent_kernel/v
о
@Adam/lstm_76/lstm_cell_76/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_76/lstm_cell_76/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Щ
 Adam/lstm_76/lstm_cell_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_76/lstm_cell_76/bias/v
Т
4Adam/lstm_76/lstm_cell_76/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_76/lstm_cell_76/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
╖,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Є+
valueш+Bх+ B▐+
є
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
╛
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
н

)layers
trainable_variables
	variables
*metrics
+layer_metrics
,layer_regularization_losses
-non_trainable_variables
regularization_losses
 
О
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
╣

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
VARIABLE_VALUEdense_92/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_92/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н

9layers
trainable_variables
	variables
:metrics
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_93/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_93/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н

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
н

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
VARIABLE_VALUElstm_76/lstm_cell_76/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_76/lstm_cell_76/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_76/lstm_cell_76/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

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
н

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
VARIABLE_VALUEAdam/dense_92/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_92/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_93/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_93/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_76/lstm_cell_76/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_76/lstm_cell_76/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_76/lstm_cell_76/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_92/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_92/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_93/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_93/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_76/lstm_cell_76/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_76/lstm_cell_76/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_76/lstm_cell_76/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Г
serving_default_input_32Placeholder*+
_output_shapes
:         *
dtype0* 
shape:         
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_32lstm_76/lstm_cell_76/kernellstm_76/lstm_cell_76/bias%lstm_76/lstm_cell_76/recurrent_kerneldense_92/kerneldense_92/biasdense_93/kerneldense_93/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2464177
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
·
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_92/kernel/Read/ReadVariableOp!dense_92/bias/Read/ReadVariableOp#dense_93/kernel/Read/ReadVariableOp!dense_93/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_76/lstm_cell_76/kernel/Read/ReadVariableOp9lstm_76/lstm_cell_76/recurrent_kernel/Read/ReadVariableOp-lstm_76/lstm_cell_76/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_92/kernel/m/Read/ReadVariableOp(Adam/dense_92/bias/m/Read/ReadVariableOp*Adam/dense_93/kernel/m/Read/ReadVariableOp(Adam/dense_93/bias/m/Read/ReadVariableOp6Adam/lstm_76/lstm_cell_76/kernel/m/Read/ReadVariableOp@Adam/lstm_76/lstm_cell_76/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_76/lstm_cell_76/bias/m/Read/ReadVariableOp*Adam/dense_92/kernel/v/Read/ReadVariableOp(Adam/dense_92/bias/v/Read/ReadVariableOp*Adam/dense_93/kernel/v/Read/ReadVariableOp(Adam/dense_93/bias/v/Read/ReadVariableOp6Adam/lstm_76/lstm_cell_76/kernel/v/Read/ReadVariableOp@Adam/lstm_76/lstm_cell_76/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_76/lstm_cell_76/bias/v/Read/ReadVariableOpConst*)
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_2466403
┼
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_92/kerneldense_92/biasdense_93/kerneldense_93/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_76/lstm_cell_76/kernel%lstm_76/lstm_cell_76/recurrent_kernellstm_76/lstm_cell_76/biastotalcountAdam/dense_92/kernel/mAdam/dense_92/bias/mAdam/dense_93/kernel/mAdam/dense_93/bias/m"Adam/lstm_76/lstm_cell_76/kernel/m,Adam/lstm_76/lstm_cell_76/recurrent_kernel/m Adam/lstm_76/lstm_cell_76/bias/mAdam/dense_92/kernel/vAdam/dense_92/bias/vAdam/dense_93/kernel/vAdam/dense_93/bias/v"Adam/lstm_76/lstm_cell_76/kernel/v,Adam/lstm_76/lstm_cell_76/recurrent_kernel/v Adam/lstm_76/lstm_cell_76/bias/v*(
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_2466497ц╧$
╢v
ь
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466285

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpZ
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
 *  А?2
ones_like/ConstД
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:          2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape╤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2├рв2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_1/ConstЕ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:          2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape╫
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2нЛ╜2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_1/GreaterEqual/y╞
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_2/ConstЕ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:          2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape╓
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╔Ф32(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_2/GreaterEqual/y╞
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_3/ConstЕ
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:          2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape╓
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Н│82(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_3/GreaterEqual/y╞
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02
split/ReadVariableOpз
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:          2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:          2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:          2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:          2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:          2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:          2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:          2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:          2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
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
strided_slice/stack_2№
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
:          2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:          2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2И
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
:          2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:          2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2И
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
:          2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:          2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:          2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:          2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2И
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
:          2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:          2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_6▌
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
┌
╚
while_cond_2463804
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2463804___redundant_placeholder05
1while_while_cond_2463804___redundant_placeholder15
1while_while_cond_2463804___redundant_placeholder25
1while_while_cond_2463804___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
зv
ъ
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2462856

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpX
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
 *  А?2
ones_like/ConstД
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:          2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape╤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2БЇо2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_1/ConstЕ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:          2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape╫
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Ём╚2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_1/GreaterEqual/y╞
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_2/ConstЕ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:          2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape╫
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╙Э┬2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_2/GreaterEqual/y╞
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_3/ConstЕ
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:          2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape╓
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2хФ)2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_3/GreaterEqual/y╞
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02
split/ReadVariableOpз
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:          2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:          2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:          2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:          2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:          2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:          2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:          2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:          2
	BiasAdd_3^
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
muld
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_1d
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
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
strided_slice/stack_2№
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
:          2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:          2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2И
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
:          2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:          2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2И
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
:          2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:          2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:          2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:          2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2И
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
:          2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:          2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_6▌
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
Т╣
й
(sequential_31_lstm_76_while_body_2462350H
Dsequential_31_lstm_76_while_sequential_31_lstm_76_while_loop_counterN
Jsequential_31_lstm_76_while_sequential_31_lstm_76_while_maximum_iterations+
'sequential_31_lstm_76_while_placeholder-
)sequential_31_lstm_76_while_placeholder_1-
)sequential_31_lstm_76_while_placeholder_2-
)sequential_31_lstm_76_while_placeholder_3G
Csequential_31_lstm_76_while_sequential_31_lstm_76_strided_slice_1_0Г
sequential_31_lstm_76_while_tensorarrayv2read_tensorlistgetitem_sequential_31_lstm_76_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_31_lstm_76_while_lstm_cell_76_split_readvariableop_resource_0:	АY
Jsequential_31_lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0:	АU
Bsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0:	 А(
$sequential_31_lstm_76_while_identity*
&sequential_31_lstm_76_while_identity_1*
&sequential_31_lstm_76_while_identity_2*
&sequential_31_lstm_76_while_identity_3*
&sequential_31_lstm_76_while_identity_4*
&sequential_31_lstm_76_while_identity_5E
Asequential_31_lstm_76_while_sequential_31_lstm_76_strided_slice_1Б
}sequential_31_lstm_76_while_tensorarrayv2read_tensorlistgetitem_sequential_31_lstm_76_tensorarrayunstack_tensorlistfromtensorY
Fsequential_31_lstm_76_while_lstm_cell_76_split_readvariableop_resource:	АW
Hsequential_31_lstm_76_while_lstm_cell_76_split_1_readvariableop_resource:	АS
@sequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource:	 АИв7sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOpв9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_1в9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_2в9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_3в=sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOpв?sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOpя
Msequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2O
Msequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_31_lstm_76_while_tensorarrayv2read_tensorlistgetitem_sequential_31_lstm_76_tensorarrayunstack_tensorlistfromtensor_0'sequential_31_lstm_76_while_placeholderVsequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02A
?sequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem═
8sequential_31/lstm_76/while/lstm_cell_76/ones_like/ShapeShape)sequential_31_lstm_76_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_31/lstm_76/while/lstm_cell_76/ones_like/Shape╣
8sequential_31/lstm_76/while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2:
8sequential_31/lstm_76/while/lstm_cell_76/ones_like/Constи
2sequential_31/lstm_76/while/lstm_cell_76/ones_likeFillAsequential_31/lstm_76/while/lstm_cell_76/ones_like/Shape:output:0Asequential_31/lstm_76/while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/ones_like╢
8sequential_31/lstm_76/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_31/lstm_76/while/lstm_cell_76/split/split_dimИ
=sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOpReadVariableOpHsequential_31_lstm_76_while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02?
=sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOp╦
.sequential_31/lstm_76/while/lstm_cell_76/splitSplitAsequential_31/lstm_76/while/lstm_cell_76/split/split_dim:output:0Esequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split20
.sequential_31/lstm_76/while/lstm_cell_76/splitЯ
/sequential_31/lstm_76/while/lstm_cell_76/MatMulMatMulFsequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_31/lstm_76/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          21
/sequential_31/lstm_76/while/lstm_cell_76/MatMulг
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_1MatMulFsequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_31/lstm_76/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_1г
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_2MatMulFsequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_31/lstm_76/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_2г
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_3MatMulFsequential_31/lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_31/lstm_76/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_3║
:sequential_31/lstm_76/while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_31/lstm_76/while/lstm_cell_76/split_1/split_dimК
?sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOpReadVariableOpJsequential_31_lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02A
?sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOp├
0sequential_31/lstm_76/while/lstm_cell_76/split_1SplitCsequential_31/lstm_76/while/lstm_cell_76/split_1/split_dim:output:0Gsequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split22
0sequential_31/lstm_76/while/lstm_cell_76/split_1Ч
0sequential_31/lstm_76/while/lstm_cell_76/BiasAddBiasAdd9sequential_31/lstm_76/while/lstm_cell_76/MatMul:product:09sequential_31/lstm_76/while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          22
0sequential_31/lstm_76/while/lstm_cell_76/BiasAddЭ
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_1BiasAdd;sequential_31/lstm_76/while/lstm_cell_76/MatMul_1:product:09sequential_31/lstm_76/while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_1Э
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_2BiasAdd;sequential_31/lstm_76/while/lstm_cell_76/MatMul_2:product:09sequential_31/lstm_76/while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_2Э
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_3BiasAdd;sequential_31/lstm_76/while/lstm_cell_76/MatMul_3:product:09sequential_31/lstm_76/while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_3¤
,sequential_31/lstm_76/while/lstm_cell_76/mulMul)sequential_31_lstm_76_while_placeholder_2;sequential_31/lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/while/lstm_cell_76/mulБ
.sequential_31/lstm_76/while/lstm_cell_76/mul_1Mul)sequential_31_lstm_76_while_placeholder_2;sequential_31/lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_1Б
.sequential_31/lstm_76/while/lstm_cell_76/mul_2Mul)sequential_31_lstm_76_while_placeholder_2;sequential_31/lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_2Б
.sequential_31/lstm_76/while/lstm_cell_76/mul_3Mul)sequential_31_lstm_76_while_placeholder_2;sequential_31/lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_3Ў
7sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOpReadVariableOpBsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype029
7sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp═
<sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack╤
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_1╤
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_2Є
6sequential_31/lstm_76/while/lstm_cell_76/strided_sliceStridedSlice?sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp:value:0Esequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack:output:0Gsequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_1:output:0Gsequential_31/lstm_76/while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_31/lstm_76/while/lstm_cell_76/strided_sliceХ
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_4MatMul0sequential_31/lstm_76/while/lstm_cell_76/mul:z:0?sequential_31/lstm_76/while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_4П
,sequential_31/lstm_76/while/lstm_cell_76/addAddV29sequential_31/lstm_76/while/lstm_cell_76/BiasAdd:output:0;sequential_31/lstm_76/while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/while/lstm_cell_76/add╙
0sequential_31/lstm_76/while/lstm_cell_76/SigmoidSigmoid0sequential_31/lstm_76/while/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          22
0sequential_31/lstm_76/while/lstm_cell_76/Sigmoid·
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_1ReadVariableOpBsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02;
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_1╤
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_1╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_2■
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1StridedSliceAsequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_1:value:0Gsequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_1:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_1Щ
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_5MatMul2sequential_31/lstm_76/while/lstm_cell_76/mul_1:z:0Asequential_31/lstm_76/while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_5Х
.sequential_31/lstm_76/while/lstm_cell_76/add_1AddV2;sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_1:output:0;sequential_31/lstm_76/while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/add_1┘
2sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_1Sigmoid2sequential_31/lstm_76/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_1№
.sequential_31/lstm_76/while/lstm_cell_76/mul_4Mul6sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_1:y:0)sequential_31_lstm_76_while_placeholder_3*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_4·
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_2ReadVariableOpBsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02;
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_2╤
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_1╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_2■
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2StridedSliceAsequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_2:value:0Gsequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_1:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_2Щ
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_6MatMul2sequential_31/lstm_76/while/lstm_cell_76/mul_2:z:0Asequential_31/lstm_76/while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_6Х
.sequential_31/lstm_76/while/lstm_cell_76/add_2AddV2;sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_2:output:0;sequential_31/lstm_76/while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/add_2╠
-sequential_31/lstm_76/while/lstm_cell_76/ReluRelu2sequential_31/lstm_76/while/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2/
-sequential_31/lstm_76/while/lstm_cell_76/ReluМ
.sequential_31/lstm_76/while/lstm_cell_76/mul_5Mul4sequential_31/lstm_76/while/lstm_cell_76/Sigmoid:y:0;sequential_31/lstm_76/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_5Г
.sequential_31/lstm_76/while/lstm_cell_76/add_3AddV22sequential_31/lstm_76/while/lstm_cell_76/mul_4:z:02sequential_31/lstm_76/while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/add_3·
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_3ReadVariableOpBsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02;
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_3╤
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2@
>sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_1╒
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_2■
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3StridedSliceAsequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_3:value:0Gsequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_1:output:0Isequential_31/lstm_76/while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_31/lstm_76/while/lstm_cell_76/strided_slice_3Щ
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_7MatMul2sequential_31/lstm_76/while/lstm_cell_76/mul_3:z:0Asequential_31/lstm_76/while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          23
1sequential_31/lstm_76/while/lstm_cell_76/MatMul_7Х
.sequential_31/lstm_76/while/lstm_cell_76/add_4AddV2;sequential_31/lstm_76/while/lstm_cell_76/BiasAdd_3:output:0;sequential_31/lstm_76/while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/add_4┘
2sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_2Sigmoid2sequential_31/lstm_76/while/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          24
2sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_2╨
/sequential_31/lstm_76/while/lstm_cell_76/Relu_1Relu2sequential_31/lstm_76/while/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          21
/sequential_31/lstm_76/while/lstm_cell_76/Relu_1Р
.sequential_31/lstm_76/while/lstm_cell_76/mul_6Mul6sequential_31/lstm_76/while/lstm_cell_76/Sigmoid_2:y:0=sequential_31/lstm_76/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          20
.sequential_31/lstm_76/while/lstm_cell_76/mul_6╬
@sequential_31/lstm_76/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_31_lstm_76_while_placeholder_1'sequential_31_lstm_76_while_placeholder2sequential_31/lstm_76/while/lstm_cell_76/mul_6:z:0*
_output_shapes
: *
element_dtype02B
@sequential_31/lstm_76/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_31/lstm_76/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_31/lstm_76/while/add/y┴
sequential_31/lstm_76/while/addAddV2'sequential_31_lstm_76_while_placeholder*sequential_31/lstm_76/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_31/lstm_76/while/addМ
#sequential_31/lstm_76/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_31/lstm_76/while/add_1/yф
!sequential_31/lstm_76/while/add_1AddV2Dsequential_31_lstm_76_while_sequential_31_lstm_76_while_loop_counter,sequential_31/lstm_76/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_31/lstm_76/while/add_1├
$sequential_31/lstm_76/while/IdentityIdentity%sequential_31/lstm_76/while/add_1:z:0!^sequential_31/lstm_76/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_31/lstm_76/while/Identityь
&sequential_31/lstm_76/while/Identity_1IdentityJsequential_31_lstm_76_while_sequential_31_lstm_76_while_maximum_iterations!^sequential_31/lstm_76/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_31/lstm_76/while/Identity_1┼
&sequential_31/lstm_76/while/Identity_2Identity#sequential_31/lstm_76/while/add:z:0!^sequential_31/lstm_76/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_31/lstm_76/while/Identity_2Є
&sequential_31/lstm_76/while/Identity_3IdentityPsequential_31/lstm_76/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_31/lstm_76/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_31/lstm_76/while/Identity_3х
&sequential_31/lstm_76/while/Identity_4Identity2sequential_31/lstm_76/while/lstm_cell_76/mul_6:z:0!^sequential_31/lstm_76/while/NoOp*
T0*'
_output_shapes
:          2(
&sequential_31/lstm_76/while/Identity_4х
&sequential_31/lstm_76/while/Identity_5Identity2sequential_31/lstm_76/while/lstm_cell_76/add_3:z:0!^sequential_31/lstm_76/while/NoOp*
T0*'
_output_shapes
:          2(
&sequential_31/lstm_76/while/Identity_5Ў
 sequential_31/lstm_76/while/NoOpNoOp8^sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp:^sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_1:^sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_2:^sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_3>^sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOp@^sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_31/lstm_76/while/NoOp"U
$sequential_31_lstm_76_while_identity-sequential_31/lstm_76/while/Identity:output:0"Y
&sequential_31_lstm_76_while_identity_1/sequential_31/lstm_76/while/Identity_1:output:0"Y
&sequential_31_lstm_76_while_identity_2/sequential_31/lstm_76/while/Identity_2:output:0"Y
&sequential_31_lstm_76_while_identity_3/sequential_31/lstm_76/while/Identity_3:output:0"Y
&sequential_31_lstm_76_while_identity_4/sequential_31/lstm_76/while/Identity_4:output:0"Y
&sequential_31_lstm_76_while_identity_5/sequential_31/lstm_76/while/Identity_5:output:0"Ж
@sequential_31_lstm_76_while_lstm_cell_76_readvariableop_resourceBsequential_31_lstm_76_while_lstm_cell_76_readvariableop_resource_0"Ц
Hsequential_31_lstm_76_while_lstm_cell_76_split_1_readvariableop_resourceJsequential_31_lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0"Т
Fsequential_31_lstm_76_while_lstm_cell_76_split_readvariableop_resourceHsequential_31_lstm_76_while_lstm_cell_76_split_readvariableop_resource_0"И
Asequential_31_lstm_76_while_sequential_31_lstm_76_strided_slice_1Csequential_31_lstm_76_while_sequential_31_lstm_76_strided_slice_1_0"А
}sequential_31_lstm_76_while_tensorarrayv2read_tensorlistgetitem_sequential_31_lstm_76_tensorarrayunstack_tensorlistfromtensorsequential_31_lstm_76_while_tensorarrayv2read_tensorlistgetitem_sequential_31_lstm_76_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2r
7sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp7sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp2v
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_19sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_12v
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_29sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_22v
9sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_39sequential_31/lstm_76/while/lstm_cell_76/ReadVariableOp_32~
=sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOp=sequential_31/lstm_76/while/lstm_cell_76/split/ReadVariableOp2В
?sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOp?sequential_31/lstm_76/while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╦

ш
lstm_76_while_cond_2464324,
(lstm_76_while_lstm_76_while_loop_counter2
.lstm_76_while_lstm_76_while_maximum_iterations
lstm_76_while_placeholder
lstm_76_while_placeholder_1
lstm_76_while_placeholder_2
lstm_76_while_placeholder_3.
*lstm_76_while_less_lstm_76_strided_slice_1E
Alstm_76_while_lstm_76_while_cond_2464324___redundant_placeholder0E
Alstm_76_while_lstm_76_while_cond_2464324___redundant_placeholder1E
Alstm_76_while_lstm_76_while_cond_2464324___redundant_placeholder2E
Alstm_76_while_lstm_76_while_cond_2464324___redundant_placeholder3
lstm_76_while_identity
Ш
lstm_76/while/LessLesslstm_76_while_placeholder*lstm_76_while_less_lstm_76_strided_slice_1*
T0*
_output_shapes
: 2
lstm_76/while/Lessu
lstm_76/while/IdentityIdentitylstm_76/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_76/while/Identity"9
lstm_76_while_identitylstm_76/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
┌
╚
while_cond_2463398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2463398___redundant_placeholder05
1while_while_cond_2463398___redundant_placeholder15
1while_while_cond_2463398___redundant_placeholder25
1while_while_cond_2463398___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
К
c
G__inference_reshape_46_layer_call_and_return_conditional_losses_2466040

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
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю═
╜
lstm_76_while_body_2464628,
(lstm_76_while_lstm_76_while_loop_counter2
.lstm_76_while_lstm_76_while_maximum_iterations
lstm_76_while_placeholder
lstm_76_while_placeholder_1
lstm_76_while_placeholder_2
lstm_76_while_placeholder_3+
'lstm_76_while_lstm_76_strided_slice_1_0g
clstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0:	АK
<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0:	АG
4lstm_76_while_lstm_cell_76_readvariableop_resource_0:	 А
lstm_76_while_identity
lstm_76_while_identity_1
lstm_76_while_identity_2
lstm_76_while_identity_3
lstm_76_while_identity_4
lstm_76_while_identity_5)
%lstm_76_while_lstm_76_strided_slice_1e
alstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensorK
8lstm_76_while_lstm_cell_76_split_readvariableop_resource:	АI
:lstm_76_while_lstm_cell_76_split_1_readvariableop_resource:	АE
2lstm_76_while_lstm_cell_76_readvariableop_resource:	 АИв)lstm_76/while/lstm_cell_76/ReadVariableOpв+lstm_76/while/lstm_cell_76/ReadVariableOp_1в+lstm_76/while/lstm_cell_76/ReadVariableOp_2в+lstm_76/while/lstm_cell_76/ReadVariableOp_3в/lstm_76/while/lstm_cell_76/split/ReadVariableOpв1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp╙
?lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_76/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0lstm_76_while_placeholderHlstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype023
1lstm_76/while/TensorArrayV2Read/TensorListGetItemг
*lstm_76/while/lstm_cell_76/ones_like/ShapeShapelstm_76_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_76/while/lstm_cell_76/ones_like/ShapeЭ
*lstm_76/while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*lstm_76/while/lstm_cell_76/ones_like/ConstЁ
$lstm_76/while/lstm_cell_76/ones_likeFill3lstm_76/while/lstm_cell_76/ones_like/Shape:output:03lstm_76/while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/ones_likeЩ
(lstm_76/while/lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2*
(lstm_76/while/lstm_cell_76/dropout/Constы
&lstm_76/while/lstm_cell_76/dropout/MulMul-lstm_76/while/lstm_cell_76/ones_like:output:01lstm_76/while/lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2(
&lstm_76/while/lstm_cell_76/dropout/Mul▒
(lstm_76/while/lstm_cell_76/dropout/ShapeShape-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_76/while/lstm_cell_76/dropout/Shapeа
?lstm_76/while/lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform1lstm_76/while/lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2∙[2A
?lstm_76/while/lstm_cell_76/dropout/random_uniform/RandomUniformл
1lstm_76/while/lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>23
1lstm_76/while/lstm_cell_76/dropout/GreaterEqual/yк
/lstm_76/while/lstm_cell_76/dropout/GreaterEqualGreaterEqualHlstm_76/while/lstm_cell_76/dropout/random_uniform/RandomUniform:output:0:lstm_76/while/lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          21
/lstm_76/while/lstm_cell_76/dropout/GreaterEqual╨
'lstm_76/while/lstm_cell_76/dropout/CastCast3lstm_76/while/lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2)
'lstm_76/while/lstm_cell_76/dropout/Castц
(lstm_76/while/lstm_cell_76/dropout/Mul_1Mul*lstm_76/while/lstm_cell_76/dropout/Mul:z:0+lstm_76/while/lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2*
(lstm_76/while/lstm_cell_76/dropout/Mul_1Э
*lstm_76/while/lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2,
*lstm_76/while/lstm_cell_76/dropout_1/Constё
(lstm_76/while/lstm_cell_76/dropout_1/MulMul-lstm_76/while/lstm_cell_76/ones_like:output:03lstm_76/while/lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2*
(lstm_76/while/lstm_cell_76/dropout_1/Mul╡
*lstm_76/while/lstm_cell_76/dropout_1/ShapeShape-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_76/while/lstm_cell_76/dropout_1/Shapeи
Alstm_76/while/lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_76/while/lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2┼│█2C
Alstm_76/while/lstm_cell_76/dropout_1/random_uniform/RandomUniformп
3lstm_76/while/lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_76/while/lstm_cell_76/dropout_1/GreaterEqual/y▓
1lstm_76/while/lstm_cell_76/dropout_1/GreaterEqualGreaterEqualJlstm_76/while/lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:0<lstm_76/while/lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          23
1lstm_76/while/lstm_cell_76/dropout_1/GreaterEqual╓
)lstm_76/while/lstm_cell_76/dropout_1/CastCast5lstm_76/while/lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2+
)lstm_76/while/lstm_cell_76/dropout_1/Castю
*lstm_76/while/lstm_cell_76/dropout_1/Mul_1Mul,lstm_76/while/lstm_cell_76/dropout_1/Mul:z:0-lstm_76/while/lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2,
*lstm_76/while/lstm_cell_76/dropout_1/Mul_1Э
*lstm_76/while/lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2,
*lstm_76/while/lstm_cell_76/dropout_2/Constё
(lstm_76/while/lstm_cell_76/dropout_2/MulMul-lstm_76/while/lstm_cell_76/ones_like:output:03lstm_76/while/lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2*
(lstm_76/while/lstm_cell_76/dropout_2/Mul╡
*lstm_76/while/lstm_cell_76/dropout_2/ShapeShape-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_76/while/lstm_cell_76/dropout_2/Shapeз
Alstm_76/while/lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_76/while/lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2оЦ_2C
Alstm_76/while/lstm_cell_76/dropout_2/random_uniform/RandomUniformп
3lstm_76/while/lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_76/while/lstm_cell_76/dropout_2/GreaterEqual/y▓
1lstm_76/while/lstm_cell_76/dropout_2/GreaterEqualGreaterEqualJlstm_76/while/lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:0<lstm_76/while/lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          23
1lstm_76/while/lstm_cell_76/dropout_2/GreaterEqual╓
)lstm_76/while/lstm_cell_76/dropout_2/CastCast5lstm_76/while/lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2+
)lstm_76/while/lstm_cell_76/dropout_2/Castю
*lstm_76/while/lstm_cell_76/dropout_2/Mul_1Mul,lstm_76/while/lstm_cell_76/dropout_2/Mul:z:0-lstm_76/while/lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2,
*lstm_76/while/lstm_cell_76/dropout_2/Mul_1Э
*lstm_76/while/lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2,
*lstm_76/while/lstm_cell_76/dropout_3/Constё
(lstm_76/while/lstm_cell_76/dropout_3/MulMul-lstm_76/while/lstm_cell_76/ones_like:output:03lstm_76/while/lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2*
(lstm_76/while/lstm_cell_76/dropout_3/Mul╡
*lstm_76/while/lstm_cell_76/dropout_3/ShapeShape-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_76/while/lstm_cell_76/dropout_3/Shapeи
Alstm_76/while/lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_76/while/lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╟╬╬2C
Alstm_76/while/lstm_cell_76/dropout_3/random_uniform/RandomUniformп
3lstm_76/while/lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_76/while/lstm_cell_76/dropout_3/GreaterEqual/y▓
1lstm_76/while/lstm_cell_76/dropout_3/GreaterEqualGreaterEqualJlstm_76/while/lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:0<lstm_76/while/lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          23
1lstm_76/while/lstm_cell_76/dropout_3/GreaterEqual╓
)lstm_76/while/lstm_cell_76/dropout_3/CastCast5lstm_76/while/lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2+
)lstm_76/while/lstm_cell_76/dropout_3/Castю
*lstm_76/while/lstm_cell_76/dropout_3/Mul_1Mul,lstm_76/while/lstm_cell_76/dropout_3/Mul:z:0-lstm_76/while/lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2,
*lstm_76/while/lstm_cell_76/dropout_3/Mul_1Ъ
*lstm_76/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_76/while/lstm_cell_76/split/split_dim▐
/lstm_76/while/lstm_cell_76/split/ReadVariableOpReadVariableOp:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/lstm_76/while/lstm_cell_76/split/ReadVariableOpУ
 lstm_76/while/lstm_cell_76/splitSplit3lstm_76/while/lstm_cell_76/split/split_dim:output:07lstm_76/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_76/while/lstm_cell_76/splitч
!lstm_76/while/lstm_cell_76/MatMulMatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2#
!lstm_76/while/lstm_cell_76/MatMulы
#lstm_76/while/lstm_cell_76/MatMul_1MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_1ы
#lstm_76/while/lstm_cell_76/MatMul_2MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_2ы
#lstm_76/while/lstm_cell_76/MatMul_3MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_3Ю
,lstm_76/while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_76/while/lstm_cell_76/split_1/split_dimр
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOpЛ
"lstm_76/while/lstm_cell_76/split_1Split5lstm_76/while/lstm_cell_76/split_1/split_dim:output:09lstm_76/while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_76/while/lstm_cell_76/split_1▀
"lstm_76/while/lstm_cell_76/BiasAddBiasAdd+lstm_76/while/lstm_cell_76/MatMul:product:0+lstm_76/while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2$
"lstm_76/while/lstm_cell_76/BiasAddх
$lstm_76/while/lstm_cell_76/BiasAdd_1BiasAdd-lstm_76/while/lstm_cell_76/MatMul_1:product:0+lstm_76/while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_1х
$lstm_76/while/lstm_cell_76/BiasAdd_2BiasAdd-lstm_76/while/lstm_cell_76/MatMul_2:product:0+lstm_76/while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_2х
$lstm_76/while/lstm_cell_76/BiasAdd_3BiasAdd-lstm_76/while/lstm_cell_76/MatMul_3:product:0+lstm_76/while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_3─
lstm_76/while/lstm_cell_76/mulMullstm_76_while_placeholder_2,lstm_76/while/lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2 
lstm_76/while/lstm_cell_76/mul╩
 lstm_76/while/lstm_cell_76/mul_1Mullstm_76_while_placeholder_2.lstm_76/while/lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_1╩
 lstm_76/while/lstm_cell_76/mul_2Mullstm_76_while_placeholder_2.lstm_76/while/lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_2╩
 lstm_76/while/lstm_cell_76/mul_3Mullstm_76_while_placeholder_2.lstm_76/while/lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_3╠
)lstm_76/while/lstm_cell_76/ReadVariableOpReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_76/while/lstm_cell_76/ReadVariableOp▒
.lstm_76/while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_76/while/lstm_cell_76/strided_slice/stack╡
0lstm_76/while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_76/while/lstm_cell_76/strided_slice/stack_1╡
0lstm_76/while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_76/while/lstm_cell_76/strided_slice/stack_2Ю
(lstm_76/while/lstm_cell_76/strided_sliceStridedSlice1lstm_76/while/lstm_cell_76/ReadVariableOp:value:07lstm_76/while/lstm_cell_76/strided_slice/stack:output:09lstm_76/while/lstm_cell_76/strided_slice/stack_1:output:09lstm_76/while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_76/while/lstm_cell_76/strided_slice▌
#lstm_76/while/lstm_cell_76/MatMul_4MatMul"lstm_76/while/lstm_cell_76/mul:z:01lstm_76/while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_4╫
lstm_76/while/lstm_cell_76/addAddV2+lstm_76/while/lstm_cell_76/BiasAdd:output:0-lstm_76/while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2 
lstm_76/while/lstm_cell_76/addй
"lstm_76/while/lstm_cell_76/SigmoidSigmoid"lstm_76/while/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2$
"lstm_76/while/lstm_cell_76/Sigmoid╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_1ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_1╡
0lstm_76/while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_76/while/lstm_cell_76/strided_slice_1/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_1StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_1:value:09lstm_76/while/lstm_cell_76/strided_slice_1/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_1/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_1с
#lstm_76/while/lstm_cell_76/MatMul_5MatMul$lstm_76/while/lstm_cell_76/mul_1:z:03lstm_76/while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_5▌
 lstm_76/while/lstm_cell_76/add_1AddV2-lstm_76/while/lstm_cell_76/BiasAdd_1:output:0-lstm_76/while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_1п
$lstm_76/while/lstm_cell_76/Sigmoid_1Sigmoid$lstm_76/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/Sigmoid_1─
 lstm_76/while/lstm_cell_76/mul_4Mul(lstm_76/while/lstm_cell_76/Sigmoid_1:y:0lstm_76_while_placeholder_3*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_4╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_2ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_2╡
0lstm_76/while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_76/while/lstm_cell_76/strided_slice_2/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_2StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_2:value:09lstm_76/while/lstm_cell_76/strided_slice_2/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_2/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_2с
#lstm_76/while/lstm_cell_76/MatMul_6MatMul$lstm_76/while/lstm_cell_76/mul_2:z:03lstm_76/while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_6▌
 lstm_76/while/lstm_cell_76/add_2AddV2-lstm_76/while/lstm_cell_76/BiasAdd_2:output:0-lstm_76/while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_2в
lstm_76/while/lstm_cell_76/ReluRelu$lstm_76/while/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2!
lstm_76/while/lstm_cell_76/Relu╘
 lstm_76/while/lstm_cell_76/mul_5Mul&lstm_76/while/lstm_cell_76/Sigmoid:y:0-lstm_76/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_5╦
 lstm_76/while/lstm_cell_76/add_3AddV2$lstm_76/while/lstm_cell_76/mul_4:z:0$lstm_76/while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_3╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_3ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_3╡
0lstm_76/while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_76/while/lstm_cell_76/strided_slice_3/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_3StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_3:value:09lstm_76/while/lstm_cell_76/strided_slice_3/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_3/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_3с
#lstm_76/while/lstm_cell_76/MatMul_7MatMul$lstm_76/while/lstm_cell_76/mul_3:z:03lstm_76/while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_7▌
 lstm_76/while/lstm_cell_76/add_4AddV2-lstm_76/while/lstm_cell_76/BiasAdd_3:output:0-lstm_76/while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_4п
$lstm_76/while/lstm_cell_76/Sigmoid_2Sigmoid$lstm_76/while/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/Sigmoid_2ж
!lstm_76/while/lstm_cell_76/Relu_1Relu$lstm_76/while/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2#
!lstm_76/while/lstm_cell_76/Relu_1╪
 lstm_76/while/lstm_cell_76/mul_6Mul(lstm_76/while/lstm_cell_76/Sigmoid_2:y:0/lstm_76/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_6И
2lstm_76/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_76_while_placeholder_1lstm_76_while_placeholder$lstm_76/while/lstm_cell_76/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_76/while/TensorArrayV2Write/TensorListSetIteml
lstm_76/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_76/while/add/yЙ
lstm_76/while/addAddV2lstm_76_while_placeholderlstm_76/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_76/while/addp
lstm_76/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_76/while/add_1/yЮ
lstm_76/while/add_1AddV2(lstm_76_while_lstm_76_while_loop_counterlstm_76/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_76/while/add_1Л
lstm_76/while/IdentityIdentitylstm_76/while/add_1:z:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identityж
lstm_76/while/Identity_1Identity.lstm_76_while_lstm_76_while_maximum_iterations^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_1Н
lstm_76/while/Identity_2Identitylstm_76/while/add:z:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_2║
lstm_76/while/Identity_3IdentityBlstm_76/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_3н
lstm_76/while/Identity_4Identity$lstm_76/while/lstm_cell_76/mul_6:z:0^lstm_76/while/NoOp*
T0*'
_output_shapes
:          2
lstm_76/while/Identity_4н
lstm_76/while/Identity_5Identity$lstm_76/while/lstm_cell_76/add_3:z:0^lstm_76/while/NoOp*
T0*'
_output_shapes
:          2
lstm_76/while/Identity_5Ж
lstm_76/while/NoOpNoOp*^lstm_76/while/lstm_cell_76/ReadVariableOp,^lstm_76/while/lstm_cell_76/ReadVariableOp_1,^lstm_76/while/lstm_cell_76/ReadVariableOp_2,^lstm_76/while/lstm_cell_76/ReadVariableOp_30^lstm_76/while/lstm_cell_76/split/ReadVariableOp2^lstm_76/while/lstm_cell_76/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_76/while/NoOp"9
lstm_76_while_identitylstm_76/while/Identity:output:0"=
lstm_76_while_identity_1!lstm_76/while/Identity_1:output:0"=
lstm_76_while_identity_2!lstm_76/while/Identity_2:output:0"=
lstm_76_while_identity_3!lstm_76/while/Identity_3:output:0"=
lstm_76_while_identity_4!lstm_76/while/Identity_4:output:0"=
lstm_76_while_identity_5!lstm_76/while/Identity_5:output:0"P
%lstm_76_while_lstm_76_strided_slice_1'lstm_76_while_lstm_76_strided_slice_1_0"j
2lstm_76_while_lstm_cell_76_readvariableop_resource4lstm_76_while_lstm_cell_76_readvariableop_resource_0"z
:lstm_76_while_lstm_cell_76_split_1_readvariableop_resource<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0"v
8lstm_76_while_lstm_cell_76_split_readvariableop_resource:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0"╚
alstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensorclstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)lstm_76/while/lstm_cell_76/ReadVariableOp)lstm_76/while/lstm_cell_76/ReadVariableOp2Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_1+lstm_76/while/lstm_cell_76/ReadVariableOp_12Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_2+lstm_76/while/lstm_cell_76/ReadVariableOp_22Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_3+lstm_76/while/lstm_cell_76/ReadVariableOp_32b
/lstm_76/while/lstm_cell_76/split/ReadVariableOp/lstm_76/while/lstm_cell_76/split/ReadVariableOp2f
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
┌
╚
while_cond_2464980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2464980___redundant_placeholder05
1while_while_cond_2464980___redundant_placeholder15
1while_while_cond_2464980___redundant_placeholder25
1while_while_cond_2464980___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
є
Ч
*__inference_dense_92_layer_call_fn_2465980

inputs
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_24635512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
№▓
е	
while_body_2465256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeЙ
 while/lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2"
 while/lstm_cell_76/dropout/Const╦
while/lstm_cell_76/dropout/MulMul%while/lstm_cell_76/ones_like:output:0)while/lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2 
while/lstm_cell_76/dropout/MulЩ
 while/lstm_cell_76/dropout/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_76/dropout/ShapeК
7while/lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2№№─29
7while/lstm_cell_76/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_76/dropout/GreaterEqual/yК
'while/lstm_cell_76/dropout/GreaterEqualGreaterEqual@while/lstm_cell_76/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2)
'while/lstm_cell_76/dropout/GreaterEqual╕
while/lstm_cell_76/dropout/CastCast+while/lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2!
while/lstm_cell_76/dropout/Cast╞
 while/lstm_cell_76/dropout/Mul_1Mul"while/lstm_cell_76/dropout/Mul:z:0#while/lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout/Mul_1Н
"while/lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_1/Const╤
 while/lstm_cell_76/dropout_1/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_1/MulЭ
"while/lstm_cell_76/dropout_1/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_1/ShapeР
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2ИУ╠2;
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_1/GreaterEqual/yТ
)while/lstm_cell_76/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_1/GreaterEqual╛
!while/lstm_cell_76/dropout_1/CastCast-while/lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_1/Cast╬
"while/lstm_cell_76/dropout_1/Mul_1Mul$while/lstm_cell_76/dropout_1/Mul:z:0%while/lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_1/Mul_1Н
"while/lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_2/Const╤
 while/lstm_cell_76/dropout_2/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_2/MulЭ
"while/lstm_cell_76/dropout_2/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_2/ShapeР
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2┼╟·2;
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_2/GreaterEqual/yТ
)while/lstm_cell_76/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_2/GreaterEqual╛
!while/lstm_cell_76/dropout_2/CastCast-while/lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_2/Cast╬
"while/lstm_cell_76/dropout_2/Mul_1Mul$while/lstm_cell_76/dropout_2/Mul:z:0%while/lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_2/Mul_1Н
"while/lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_3/Const╤
 while/lstm_cell_76/dropout_3/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_3/MulЭ
"while/lstm_cell_76/dropout_3/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_3/ShapeР
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Ц╫╬2;
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_3/GreaterEqual/yТ
)while/lstm_cell_76/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_3/GreaterEqual╛
!while/lstm_cell_76/dropout_3/CastCast-while/lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_3/Cast╬
"while/lstm_cell_76/dropout_3/Mul_1Mul$while/lstm_cell_76/dropout_3/Mul:z:0%while/lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_3/Mul_1К
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3д
while/lstm_cell_76/mulMulwhile_placeholder_2$while/lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulк
while/lstm_cell_76/mul_1Mulwhile_placeholder_2&while/lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1к
while/lstm_cell_76/mul_2Mulwhile_placeholder_2&while/lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2к
while/lstm_cell_76/mul_3Mulwhile_placeholder_2&while/lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
█╧
и
D__inference_lstm_76_layer_call_and_return_conditional_losses_2463970

inputs=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
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
:         2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like}
lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout/Const│
lstm_cell_76/dropout/MulMullstm_cell_76/ones_like:output:0#lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/MulЗ
lstm_cell_76/dropout/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout/Shape°
1lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╥Ў╣23
1lstm_cell_76/dropout/random_uniform/RandomUniformП
#lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_76/dropout/GreaterEqual/yЄ
!lstm_cell_76/dropout/GreaterEqualGreaterEqual:lstm_cell_76/dropout/random_uniform/RandomUniform:output:0,lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2#
!lstm_cell_76/dropout/GreaterEqualж
lstm_cell_76/dropout/CastCast%lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout/Castо
lstm_cell_76/dropout/Mul_1Mullstm_cell_76/dropout/Mul:z:0lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/Mul_1Б
lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_1/Const╣
lstm_cell_76/dropout_1/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/MulЛ
lstm_cell_76/dropout_1/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_1/Shape■
3lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2шО┌25
3lstm_cell_76/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_1/GreaterEqual/y·
#lstm_cell_76/dropout_1/GreaterEqualGreaterEqual<lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_1/GreaterEqualм
lstm_cell_76/dropout_1/CastCast'lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Cast╢
lstm_cell_76/dropout_1/Mul_1Mullstm_cell_76/dropout_1/Mul:z:0lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Mul_1Б
lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_2/Const╣
lstm_cell_76/dropout_2/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/MulЛ
lstm_cell_76/dropout_2/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_2/Shape¤
3lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Ф┌25
3lstm_cell_76/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_2/GreaterEqual/y·
#lstm_cell_76/dropout_2/GreaterEqualGreaterEqual<lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_2/GreaterEqualм
lstm_cell_76/dropout_2/CastCast'lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Cast╢
lstm_cell_76/dropout_2/Mul_1Mullstm_cell_76/dropout_2/Mul:z:0lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Mul_1Б
lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_3/Const╣
lstm_cell_76/dropout_3/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/MulЛ
lstm_cell_76/dropout_3/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_3/Shape■
3lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2шТ┬25
3lstm_cell_76/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_3/GreaterEqual/y·
#lstm_cell_76/dropout_3/GreaterEqualGreaterEqual<lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_3/GreaterEqualм
lstm_cell_76/dropout_3/CastCast'lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Cast╢
lstm_cell_76/dropout_3/Mul_1Mullstm_cell_76/dropout_3/Mul:z:0lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Mul_1~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3Н
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulУ
lstm_cell_76/mul_1Mulzeros:output:0 lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1У
lstm_cell_76/mul_2Mulzeros:output:0 lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2У
lstm_cell_76/mul_3Mulzeros:output:0 lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2463805*
condR
while_cond_2463804*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╕
ў
.__inference_lstm_cell_76_layer_call_fn_2466074

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24626232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          2

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
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
и
╢
)__inference_lstm_76_layer_call_fn_2464871

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24639702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
═B
у
 __inference__traced_save_2466403
file_prefix.
*savev2_dense_92_kernel_read_readvariableop,
(savev2_dense_92_bias_read_readvariableop.
*savev2_dense_93_kernel_read_readvariableop,
(savev2_dense_93_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_76_lstm_cell_76_kernel_read_readvariableopD
@savev2_lstm_76_lstm_cell_76_recurrent_kernel_read_readvariableop8
4savev2_lstm_76_lstm_cell_76_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_92_kernel_m_read_readvariableop3
/savev2_adam_dense_92_bias_m_read_readvariableop5
1savev2_adam_dense_93_kernel_m_read_readvariableop3
/savev2_adam_dense_93_bias_m_read_readvariableopA
=savev2_adam_lstm_76_lstm_cell_76_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_76_lstm_cell_76_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_76_lstm_cell_76_bias_m_read_readvariableop5
1savev2_adam_dense_92_kernel_v_read_readvariableop3
/savev2_adam_dense_92_bias_v_read_readvariableop5
1savev2_adam_dense_93_kernel_v_read_readvariableop3
/savev2_adam_dense_93_bias_v_read_readvariableopA
=savev2_adam_lstm_76_lstm_cell_76_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_76_lstm_cell_76_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_76_lstm_cell_76_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╨
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
value╪B╒B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names┬
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▄
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_92_kernel_read_readvariableop(savev2_dense_92_bias_read_readvariableop*savev2_dense_93_kernel_read_readvariableop(savev2_dense_93_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_76_lstm_cell_76_kernel_read_readvariableop@savev2_lstm_76_lstm_cell_76_recurrent_kernel_read_readvariableop4savev2_lstm_76_lstm_cell_76_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_92_kernel_m_read_readvariableop/savev2_adam_dense_92_bias_m_read_readvariableop1savev2_adam_dense_93_kernel_m_read_readvariableop/savev2_adam_dense_93_bias_m_read_readvariableop=savev2_adam_lstm_76_lstm_cell_76_kernel_m_read_readvariableopGsavev2_adam_lstm_76_lstm_cell_76_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_76_lstm_cell_76_bias_m_read_readvariableop1savev2_adam_dense_92_kernel_v_read_readvariableop/savev2_adam_dense_92_bias_v_read_readvariableop1savev2_adam_dense_93_kernel_v_read_readvariableop/savev2_adam_dense_93_bias_v_read_readvariableop=savev2_adam_lstm_76_lstm_cell_76_kernel_v_read_readvariableopGsavev2_adam_lstm_76_lstm_cell_76_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_76_lstm_cell_76_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*▐
_input_shapes╠
╔: :  : : :: : : : : :	А:	 А:А: : :  : : ::	А:	 А:А:  : : ::	А:	 А:А: 2(
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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:
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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:$ 

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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:

_output_shapes
: 
ъ	
к
/__inference_sequential_31_layer_call_fn_2464070
input_32
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_31_layer_call_and_return_conditional_losses_24640342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
сб
и
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465664

inputs=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
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
:         2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3О
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulТ
lstm_cell_76/mul_1Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1Т
lstm_cell_76/mul_2Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2Т
lstm_cell_76/mul_3Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2465531*
condR
while_cond_2465530*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Г
Ў
E__inference_dense_92_layer_call_and_return_conditional_losses_2465991

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▌
╦
__inference_loss_fn_1_2466296Y
Flstm_76_lstm_cell_76_kernel_regularizer_square_readvariableop_resource:	А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpЖ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_76_lstm_cell_76_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muly
IdentityIdentity/lstm_76/lstm_cell_76/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityО
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp
ЪR
╔
D__inference_lstm_76_layer_call_and_return_conditional_losses_2462712

inputs'
lstm_cell_76_2462624:	А#
lstm_cell_76_2462626:	А'
lstm_cell_76_2462628:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpв$lstm_cell_76/StatefulPartitionedCallвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2б
$lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_76_2462624lstm_cell_76_2462626lstm_cell_76_2462628*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24626232&
$lstm_cell_76/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_76_2462624lstm_cell_76_2462626lstm_cell_76_2462628*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2462637*
condR
while_cond_2462636*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime╘
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_76_2462624*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity╜
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_76/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_76/StatefulPartitionedCall$lstm_cell_76/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ё
А
(sequential_31_lstm_76_while_cond_2462349H
Dsequential_31_lstm_76_while_sequential_31_lstm_76_while_loop_counterN
Jsequential_31_lstm_76_while_sequential_31_lstm_76_while_maximum_iterations+
'sequential_31_lstm_76_while_placeholder-
)sequential_31_lstm_76_while_placeholder_1-
)sequential_31_lstm_76_while_placeholder_2-
)sequential_31_lstm_76_while_placeholder_3J
Fsequential_31_lstm_76_while_less_sequential_31_lstm_76_strided_slice_1a
]sequential_31_lstm_76_while_sequential_31_lstm_76_while_cond_2462349___redundant_placeholder0a
]sequential_31_lstm_76_while_sequential_31_lstm_76_while_cond_2462349___redundant_placeholder1a
]sequential_31_lstm_76_while_sequential_31_lstm_76_while_cond_2462349___redundant_placeholder2a
]sequential_31_lstm_76_while_sequential_31_lstm_76_while_cond_2462349___redundant_placeholder3(
$sequential_31_lstm_76_while_identity
▐
 sequential_31/lstm_76/while/LessLess'sequential_31_lstm_76_while_placeholderFsequential_31_lstm_76_while_less_sequential_31_lstm_76_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_31/lstm_76/while/LessЯ
$sequential_31/lstm_76/while/IdentityIdentity$sequential_31/lstm_76/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_31/lstm_76/while/Identity"U
$sequential_31_lstm_76_while_identity-sequential_31/lstm_76/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
¤Ф
╜
lstm_76_while_body_2464325,
(lstm_76_while_lstm_76_while_loop_counter2
.lstm_76_while_lstm_76_while_maximum_iterations
lstm_76_while_placeholder
lstm_76_while_placeholder_1
lstm_76_while_placeholder_2
lstm_76_while_placeholder_3+
'lstm_76_while_lstm_76_strided_slice_1_0g
clstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0:	АK
<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0:	АG
4lstm_76_while_lstm_cell_76_readvariableop_resource_0:	 А
lstm_76_while_identity
lstm_76_while_identity_1
lstm_76_while_identity_2
lstm_76_while_identity_3
lstm_76_while_identity_4
lstm_76_while_identity_5)
%lstm_76_while_lstm_76_strided_slice_1e
alstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensorK
8lstm_76_while_lstm_cell_76_split_readvariableop_resource:	АI
:lstm_76_while_lstm_cell_76_split_1_readvariableop_resource:	АE
2lstm_76_while_lstm_cell_76_readvariableop_resource:	 АИв)lstm_76/while/lstm_cell_76/ReadVariableOpв+lstm_76/while/lstm_cell_76/ReadVariableOp_1в+lstm_76/while/lstm_cell_76/ReadVariableOp_2в+lstm_76/while/lstm_cell_76/ReadVariableOp_3в/lstm_76/while/lstm_cell_76/split/ReadVariableOpв1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp╙
?lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?lstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_76/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0lstm_76_while_placeholderHlstm_76/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype023
1lstm_76/while/TensorArrayV2Read/TensorListGetItemг
*lstm_76/while/lstm_cell_76/ones_like/ShapeShapelstm_76_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_76/while/lstm_cell_76/ones_like/ShapeЭ
*lstm_76/while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*lstm_76/while/lstm_cell_76/ones_like/ConstЁ
$lstm_76/while/lstm_cell_76/ones_likeFill3lstm_76/while/lstm_cell_76/ones_like/Shape:output:03lstm_76/while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/ones_likeЪ
*lstm_76/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_76/while/lstm_cell_76/split/split_dim▐
/lstm_76/while/lstm_cell_76/split/ReadVariableOpReadVariableOp:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/lstm_76/while/lstm_cell_76/split/ReadVariableOpУ
 lstm_76/while/lstm_cell_76/splitSplit3lstm_76/while/lstm_cell_76/split/split_dim:output:07lstm_76/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_76/while/lstm_cell_76/splitч
!lstm_76/while/lstm_cell_76/MatMulMatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2#
!lstm_76/while/lstm_cell_76/MatMulы
#lstm_76/while/lstm_cell_76/MatMul_1MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_1ы
#lstm_76/while/lstm_cell_76/MatMul_2MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_2ы
#lstm_76/while/lstm_cell_76/MatMul_3MatMul8lstm_76/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_76/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_3Ю
,lstm_76/while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_76/while/lstm_cell_76/split_1/split_dimр
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOpЛ
"lstm_76/while/lstm_cell_76/split_1Split5lstm_76/while/lstm_cell_76/split_1/split_dim:output:09lstm_76/while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_76/while/lstm_cell_76/split_1▀
"lstm_76/while/lstm_cell_76/BiasAddBiasAdd+lstm_76/while/lstm_cell_76/MatMul:product:0+lstm_76/while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2$
"lstm_76/while/lstm_cell_76/BiasAddх
$lstm_76/while/lstm_cell_76/BiasAdd_1BiasAdd-lstm_76/while/lstm_cell_76/MatMul_1:product:0+lstm_76/while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_1х
$lstm_76/while/lstm_cell_76/BiasAdd_2BiasAdd-lstm_76/while/lstm_cell_76/MatMul_2:product:0+lstm_76/while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_2х
$lstm_76/while/lstm_cell_76/BiasAdd_3BiasAdd-lstm_76/while/lstm_cell_76/MatMul_3:product:0+lstm_76/while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/BiasAdd_3┼
lstm_76/while/lstm_cell_76/mulMullstm_76_while_placeholder_2-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2 
lstm_76/while/lstm_cell_76/mul╔
 lstm_76/while/lstm_cell_76/mul_1Mullstm_76_while_placeholder_2-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_1╔
 lstm_76/while/lstm_cell_76/mul_2Mullstm_76_while_placeholder_2-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_2╔
 lstm_76/while/lstm_cell_76/mul_3Mullstm_76_while_placeholder_2-lstm_76/while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_3╠
)lstm_76/while/lstm_cell_76/ReadVariableOpReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_76/while/lstm_cell_76/ReadVariableOp▒
.lstm_76/while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_76/while/lstm_cell_76/strided_slice/stack╡
0lstm_76/while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_76/while/lstm_cell_76/strided_slice/stack_1╡
0lstm_76/while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_76/while/lstm_cell_76/strided_slice/stack_2Ю
(lstm_76/while/lstm_cell_76/strided_sliceStridedSlice1lstm_76/while/lstm_cell_76/ReadVariableOp:value:07lstm_76/while/lstm_cell_76/strided_slice/stack:output:09lstm_76/while/lstm_cell_76/strided_slice/stack_1:output:09lstm_76/while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_76/while/lstm_cell_76/strided_slice▌
#lstm_76/while/lstm_cell_76/MatMul_4MatMul"lstm_76/while/lstm_cell_76/mul:z:01lstm_76/while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_4╫
lstm_76/while/lstm_cell_76/addAddV2+lstm_76/while/lstm_cell_76/BiasAdd:output:0-lstm_76/while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2 
lstm_76/while/lstm_cell_76/addй
"lstm_76/while/lstm_cell_76/SigmoidSigmoid"lstm_76/while/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2$
"lstm_76/while/lstm_cell_76/Sigmoid╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_1ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_1╡
0lstm_76/while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_76/while/lstm_cell_76/strided_slice_1/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_1/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_1StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_1:value:09lstm_76/while/lstm_cell_76/strided_slice_1/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_1/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_1с
#lstm_76/while/lstm_cell_76/MatMul_5MatMul$lstm_76/while/lstm_cell_76/mul_1:z:03lstm_76/while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_5▌
 lstm_76/while/lstm_cell_76/add_1AddV2-lstm_76/while/lstm_cell_76/BiasAdd_1:output:0-lstm_76/while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_1п
$lstm_76/while/lstm_cell_76/Sigmoid_1Sigmoid$lstm_76/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/Sigmoid_1─
 lstm_76/while/lstm_cell_76/mul_4Mul(lstm_76/while/lstm_cell_76/Sigmoid_1:y:0lstm_76_while_placeholder_3*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_4╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_2ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_2╡
0lstm_76/while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_76/while/lstm_cell_76/strided_slice_2/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_2/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_2StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_2:value:09lstm_76/while/lstm_cell_76/strided_slice_2/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_2/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_2с
#lstm_76/while/lstm_cell_76/MatMul_6MatMul$lstm_76/while/lstm_cell_76/mul_2:z:03lstm_76/while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_6▌
 lstm_76/while/lstm_cell_76/add_2AddV2-lstm_76/while/lstm_cell_76/BiasAdd_2:output:0-lstm_76/while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_2в
lstm_76/while/lstm_cell_76/ReluRelu$lstm_76/while/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2!
lstm_76/while/lstm_cell_76/Relu╘
 lstm_76/while/lstm_cell_76/mul_5Mul&lstm_76/while/lstm_cell_76/Sigmoid:y:0-lstm_76/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_5╦
 lstm_76/while/lstm_cell_76/add_3AddV2$lstm_76/while/lstm_cell_76/mul_4:z:0$lstm_76/while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_3╨
+lstm_76/while/lstm_cell_76/ReadVariableOp_3ReadVariableOp4lstm_76_while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_76/while/lstm_cell_76/ReadVariableOp_3╡
0lstm_76/while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_76/while/lstm_cell_76/strided_slice_3/stack╣
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_1╣
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_76/while/lstm_cell_76/strided_slice_3/stack_2к
*lstm_76/while/lstm_cell_76/strided_slice_3StridedSlice3lstm_76/while/lstm_cell_76/ReadVariableOp_3:value:09lstm_76/while/lstm_cell_76/strided_slice_3/stack:output:0;lstm_76/while/lstm_cell_76/strided_slice_3/stack_1:output:0;lstm_76/while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_76/while/lstm_cell_76/strided_slice_3с
#lstm_76/while/lstm_cell_76/MatMul_7MatMul$lstm_76/while/lstm_cell_76/mul_3:z:03lstm_76/while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2%
#lstm_76/while/lstm_cell_76/MatMul_7▌
 lstm_76/while/lstm_cell_76/add_4AddV2-lstm_76/while/lstm_cell_76/BiasAdd_3:output:0-lstm_76/while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/add_4п
$lstm_76/while/lstm_cell_76/Sigmoid_2Sigmoid$lstm_76/while/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2&
$lstm_76/while/lstm_cell_76/Sigmoid_2ж
!lstm_76/while/lstm_cell_76/Relu_1Relu$lstm_76/while/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2#
!lstm_76/while/lstm_cell_76/Relu_1╪
 lstm_76/while/lstm_cell_76/mul_6Mul(lstm_76/while/lstm_cell_76/Sigmoid_2:y:0/lstm_76/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2"
 lstm_76/while/lstm_cell_76/mul_6И
2lstm_76/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_76_while_placeholder_1lstm_76_while_placeholder$lstm_76/while/lstm_cell_76/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_76/while/TensorArrayV2Write/TensorListSetIteml
lstm_76/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_76/while/add/yЙ
lstm_76/while/addAddV2lstm_76_while_placeholderlstm_76/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_76/while/addp
lstm_76/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_76/while/add_1/yЮ
lstm_76/while/add_1AddV2(lstm_76_while_lstm_76_while_loop_counterlstm_76/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_76/while/add_1Л
lstm_76/while/IdentityIdentitylstm_76/while/add_1:z:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identityж
lstm_76/while/Identity_1Identity.lstm_76_while_lstm_76_while_maximum_iterations^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_1Н
lstm_76/while/Identity_2Identitylstm_76/while/add:z:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_2║
lstm_76/while/Identity_3IdentityBlstm_76/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_76/while/NoOp*
T0*
_output_shapes
: 2
lstm_76/while/Identity_3н
lstm_76/while/Identity_4Identity$lstm_76/while/lstm_cell_76/mul_6:z:0^lstm_76/while/NoOp*
T0*'
_output_shapes
:          2
lstm_76/while/Identity_4н
lstm_76/while/Identity_5Identity$lstm_76/while/lstm_cell_76/add_3:z:0^lstm_76/while/NoOp*
T0*'
_output_shapes
:          2
lstm_76/while/Identity_5Ж
lstm_76/while/NoOpNoOp*^lstm_76/while/lstm_cell_76/ReadVariableOp,^lstm_76/while/lstm_cell_76/ReadVariableOp_1,^lstm_76/while/lstm_cell_76/ReadVariableOp_2,^lstm_76/while/lstm_cell_76/ReadVariableOp_30^lstm_76/while/lstm_cell_76/split/ReadVariableOp2^lstm_76/while/lstm_cell_76/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_76/while/NoOp"9
lstm_76_while_identitylstm_76/while/Identity:output:0"=
lstm_76_while_identity_1!lstm_76/while/Identity_1:output:0"=
lstm_76_while_identity_2!lstm_76/while/Identity_2:output:0"=
lstm_76_while_identity_3!lstm_76/while/Identity_3:output:0"=
lstm_76_while_identity_4!lstm_76/while/Identity_4:output:0"=
lstm_76_while_identity_5!lstm_76/while/Identity_5:output:0"P
%lstm_76_while_lstm_76_strided_slice_1'lstm_76_while_lstm_76_strided_slice_1_0"j
2lstm_76_while_lstm_cell_76_readvariableop_resource4lstm_76_while_lstm_cell_76_readvariableop_resource_0"z
:lstm_76_while_lstm_cell_76_split_1_readvariableop_resource<lstm_76_while_lstm_cell_76_split_1_readvariableop_resource_0"v
8lstm_76_while_lstm_cell_76_split_readvariableop_resource:lstm_76_while_lstm_cell_76_split_readvariableop_resource_0"╚
alstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensorclstm_76_while_tensorarrayv2read_tensorlistgetitem_lstm_76_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)lstm_76/while/lstm_cell_76/ReadVariableOp)lstm_76/while/lstm_cell_76/ReadVariableOp2Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_1+lstm_76/while/lstm_cell_76/ReadVariableOp_12Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_2+lstm_76/while/lstm_cell_76/ReadVariableOp_22Z
+lstm_76/while/lstm_cell_76/ReadVariableOp_3+lstm_76/while/lstm_cell_76/ReadVariableOp_32b
/lstm_76/while/lstm_cell_76/split/ReadVariableOp/lstm_76/while/lstm_cell_76/split/ReadVariableOp2f
1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp1lstm_76/while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
щ%
ъ
while_body_2462637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_76_2462661_0:	А+
while_lstm_cell_76_2462663_0:	А/
while_lstm_cell_76_2462665_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_76_2462661:	А)
while_lstm_cell_76_2462663:	А-
while_lstm_cell_76_2462665:	 АИв*while/lstm_cell_76/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
*while/lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_76_2462661_0while_lstm_cell_76_2462663_0while_lstm_cell_76_2462665_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24626232,
*while/lstm_cell_76/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_76/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_76/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_76/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_76/StatefulPartitionedCall*"
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
while_lstm_cell_76_2462661while_lstm_cell_76_2462661_0":
while_lstm_cell_76_2462663while_lstm_cell_76_2462663_0":
while_lstm_cell_76_2462665while_lstm_cell_76_2462665_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_76/StatefulPartitionedCall*while/lstm_cell_76/StatefulPartitionedCall: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
·▓
е	
while_body_2463805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeЙ
 while/lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2"
 while/lstm_cell_76/dropout/Const╦
while/lstm_cell_76/dropout/MulMul%while/lstm_cell_76/ones_like:output:0)while/lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2 
while/lstm_cell_76/dropout/MulЩ
 while/lstm_cell_76/dropout/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_76/dropout/ShapeК
7while/lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2ўуЄ29
7while/lstm_cell_76/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_76/dropout/GreaterEqual/yК
'while/lstm_cell_76/dropout/GreaterEqualGreaterEqual@while/lstm_cell_76/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2)
'while/lstm_cell_76/dropout/GreaterEqual╕
while/lstm_cell_76/dropout/CastCast+while/lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2!
while/lstm_cell_76/dropout/Cast╞
 while/lstm_cell_76/dropout/Mul_1Mul"while/lstm_cell_76/dropout/Mul:z:0#while/lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout/Mul_1Н
"while/lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_1/Const╤
 while/lstm_cell_76/dropout_1/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_1/MulЭ
"while/lstm_cell_76/dropout_1/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_1/ShapeП
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╪╤G2;
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_1/GreaterEqual/yТ
)while/lstm_cell_76/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_1/GreaterEqual╛
!while/lstm_cell_76/dropout_1/CastCast-while/lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_1/Cast╬
"while/lstm_cell_76/dropout_1/Mul_1Mul$while/lstm_cell_76/dropout_1/Mul:z:0%while/lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_1/Mul_1Н
"while/lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_2/Const╤
 while/lstm_cell_76/dropout_2/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_2/MulЭ
"while/lstm_cell_76/dropout_2/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_2/ShapeП
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2■·J2;
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_2/GreaterEqual/yТ
)while/lstm_cell_76/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_2/GreaterEqual╛
!while/lstm_cell_76/dropout_2/CastCast-while/lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_2/Cast╬
"while/lstm_cell_76/dropout_2/Mul_1Mul$while/lstm_cell_76/dropout_2/Mul:z:0%while/lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_2/Mul_1Н
"while/lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_3/Const╤
 while/lstm_cell_76/dropout_3/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_3/MulЭ
"while/lstm_cell_76/dropout_3/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_3/ShapeР
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╥╬╞2;
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_3/GreaterEqual/yТ
)while/lstm_cell_76/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_3/GreaterEqual╛
!while/lstm_cell_76/dropout_3/CastCast-while/lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_3/Cast╬
"while/lstm_cell_76/dropout_3/Mul_1Mul$while/lstm_cell_76/dropout_3/Mul:z:0%while/lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_3/Mul_1К
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3д
while/lstm_cell_76/mulMulwhile_placeholder_2$while/lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulк
while/lstm_cell_76/mul_1Mulwhile_placeholder_2&while/lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1к
while/lstm_cell_76/mul_2Mulwhile_placeholder_2&while/lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2к
while/lstm_cell_76/mul_3Mulwhile_placeholder_2&while/lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
№▓
е	
while_body_2465806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeЙ
 while/lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2"
 while/lstm_cell_76/dropout/Const╦
while/lstm_cell_76/dropout/MulMul%while/lstm_cell_76/ones_like:output:0)while/lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2 
while/lstm_cell_76/dropout/MulЩ
 while/lstm_cell_76/dropout/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_76/dropout/ShapeК
7while/lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2┼█Ш29
7while/lstm_cell_76/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_76/dropout/GreaterEqual/yК
'while/lstm_cell_76/dropout/GreaterEqualGreaterEqual@while/lstm_cell_76/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2)
'while/lstm_cell_76/dropout/GreaterEqual╕
while/lstm_cell_76/dropout/CastCast+while/lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2!
while/lstm_cell_76/dropout/Cast╞
 while/lstm_cell_76/dropout/Mul_1Mul"while/lstm_cell_76/dropout/Mul:z:0#while/lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout/Mul_1Н
"while/lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_1/Const╤
 while/lstm_cell_76/dropout_1/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_1/MulЭ
"while/lstm_cell_76/dropout_1/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_1/ShapeР
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╬Вщ2;
9while/lstm_cell_76/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_1/GreaterEqual/yТ
)while/lstm_cell_76/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_1/GreaterEqual╛
!while/lstm_cell_76/dropout_1/CastCast-while/lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_1/Cast╬
"while/lstm_cell_76/dropout_1/Mul_1Mul$while/lstm_cell_76/dropout_1/Mul:z:0%while/lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_1/Mul_1Н
"while/lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_2/Const╤
 while/lstm_cell_76/dropout_2/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_2/MulЭ
"while/lstm_cell_76/dropout_2/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_2/ShapeР
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2ъеГ2;
9while/lstm_cell_76/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_2/GreaterEqual/yТ
)while/lstm_cell_76/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_2/GreaterEqual╛
!while/lstm_cell_76/dropout_2/CastCast-while/lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_2/Cast╬
"while/lstm_cell_76/dropout_2/Mul_1Mul$while/lstm_cell_76/dropout_2/Mul:z:0%while/lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_2/Mul_1Н
"while/lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"while/lstm_cell_76/dropout_3/Const╤
 while/lstm_cell_76/dropout_3/MulMul%while/lstm_cell_76/ones_like:output:0+while/lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2"
 while/lstm_cell_76/dropout_3/MulЭ
"while/lstm_cell_76/dropout_3/ShapeShape%while/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_76/dropout_3/ShapeР
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2├╕╠2;
9while/lstm_cell_76/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_76/dropout_3/GreaterEqual/yТ
)while/lstm_cell_76/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)while/lstm_cell_76/dropout_3/GreaterEqual╛
!while/lstm_cell_76/dropout_3/CastCast-while/lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!while/lstm_cell_76/dropout_3/Cast╬
"while/lstm_cell_76/dropout_3/Mul_1Mul$while/lstm_cell_76/dropout_3/Mul:z:0%while/lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2$
"while/lstm_cell_76/dropout_3/Mul_1К
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3д
while/lstm_cell_76/mulMulwhile_placeholder_2$while/lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulк
while/lstm_cell_76/mul_1Mulwhile_placeholder_2&while/lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1к
while/lstm_cell_76/mul_2Mulwhile_placeholder_2&while/lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2к
while/lstm_cell_76/mul_3Mulwhile_placeholder_2&while/lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
░■
А	
"__inference__wrapped_model_2462499
input_32S
@sequential_31_lstm_76_lstm_cell_76_split_readvariableop_resource:	АQ
Bsequential_31_lstm_76_lstm_cell_76_split_1_readvariableop_resource:	АM
:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource:	 АG
5sequential_31_dense_92_matmul_readvariableop_resource:  D
6sequential_31_dense_92_biasadd_readvariableop_resource: G
5sequential_31_dense_93_matmul_readvariableop_resource: D
6sequential_31_dense_93_biasadd_readvariableop_resource:
identityИв-sequential_31/dense_92/BiasAdd/ReadVariableOpв,sequential_31/dense_92/MatMul/ReadVariableOpв-sequential_31/dense_93/BiasAdd/ReadVariableOpв,sequential_31/dense_93/MatMul/ReadVariableOpв1sequential_31/lstm_76/lstm_cell_76/ReadVariableOpв3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_1в3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_2в3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_3в7sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOpв9sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOpвsequential_31/lstm_76/whiler
sequential_31/lstm_76/ShapeShapeinput_32*
T0*
_output_shapes
:2
sequential_31/lstm_76/Shapeа
)sequential_31/lstm_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_31/lstm_76/strided_slice/stackд
+sequential_31/lstm_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_31/lstm_76/strided_slice/stack_1д
+sequential_31/lstm_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_31/lstm_76/strided_slice/stack_2ц
#sequential_31/lstm_76/strided_sliceStridedSlice$sequential_31/lstm_76/Shape:output:02sequential_31/lstm_76/strided_slice/stack:output:04sequential_31/lstm_76/strided_slice/stack_1:output:04sequential_31/lstm_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_31/lstm_76/strided_sliceИ
!sequential_31/lstm_76/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_31/lstm_76/zeros/mul/y─
sequential_31/lstm_76/zeros/mulMul,sequential_31/lstm_76/strided_slice:output:0*sequential_31/lstm_76/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_31/lstm_76/zeros/mulЛ
"sequential_31/lstm_76/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_31/lstm_76/zeros/Less/y┐
 sequential_31/lstm_76/zeros/LessLess#sequential_31/lstm_76/zeros/mul:z:0+sequential_31/lstm_76/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_31/lstm_76/zeros/LessО
$sequential_31/lstm_76/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_31/lstm_76/zeros/packed/1█
"sequential_31/lstm_76/zeros/packedPack,sequential_31/lstm_76/strided_slice:output:0-sequential_31/lstm_76/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_31/lstm_76/zeros/packedЛ
!sequential_31/lstm_76/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_31/lstm_76/zeros/Const═
sequential_31/lstm_76/zerosFill+sequential_31/lstm_76/zeros/packed:output:0*sequential_31/lstm_76/zeros/Const:output:0*
T0*'
_output_shapes
:          2
sequential_31/lstm_76/zerosМ
#sequential_31/lstm_76/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_31/lstm_76/zeros_1/mul/y╩
!sequential_31/lstm_76/zeros_1/mulMul,sequential_31/lstm_76/strided_slice:output:0,sequential_31/lstm_76/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_31/lstm_76/zeros_1/mulП
$sequential_31/lstm_76/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_31/lstm_76/zeros_1/Less/y╟
"sequential_31/lstm_76/zeros_1/LessLess%sequential_31/lstm_76/zeros_1/mul:z:0-sequential_31/lstm_76/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_31/lstm_76/zeros_1/LessТ
&sequential_31/lstm_76/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_31/lstm_76/zeros_1/packed/1с
$sequential_31/lstm_76/zeros_1/packedPack,sequential_31/lstm_76/strided_slice:output:0/sequential_31/lstm_76/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_31/lstm_76/zeros_1/packedП
#sequential_31/lstm_76/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_31/lstm_76/zeros_1/Const╒
sequential_31/lstm_76/zeros_1Fill-sequential_31/lstm_76/zeros_1/packed:output:0,sequential_31/lstm_76/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
sequential_31/lstm_76/zeros_1б
$sequential_31/lstm_76/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_31/lstm_76/transpose/perm╛
sequential_31/lstm_76/transpose	Transposeinput_32-sequential_31/lstm_76/transpose/perm:output:0*
T0*+
_output_shapes
:         2!
sequential_31/lstm_76/transposeС
sequential_31/lstm_76/Shape_1Shape#sequential_31/lstm_76/transpose:y:0*
T0*
_output_shapes
:2
sequential_31/lstm_76/Shape_1д
+sequential_31/lstm_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_31/lstm_76/strided_slice_1/stackи
-sequential_31/lstm_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_31/lstm_76/strided_slice_1/stack_1и
-sequential_31/lstm_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_31/lstm_76/strided_slice_1/stack_2Є
%sequential_31/lstm_76/strided_slice_1StridedSlice&sequential_31/lstm_76/Shape_1:output:04sequential_31/lstm_76/strided_slice_1/stack:output:06sequential_31/lstm_76/strided_slice_1/stack_1:output:06sequential_31/lstm_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_31/lstm_76/strided_slice_1▒
1sequential_31/lstm_76/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_31/lstm_76/TensorArrayV2/element_shapeК
#sequential_31/lstm_76/TensorArrayV2TensorListReserve:sequential_31/lstm_76/TensorArrayV2/element_shape:output:0.sequential_31/lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_31/lstm_76/TensorArrayV2ы
Ksequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2M
Ksequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_31/lstm_76/transpose:y:0Tsequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensorд
+sequential_31/lstm_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_31/lstm_76/strided_slice_2/stackи
-sequential_31/lstm_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_31/lstm_76/strided_slice_2/stack_1и
-sequential_31/lstm_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_31/lstm_76/strided_slice_2/stack_2А
%sequential_31/lstm_76/strided_slice_2StridedSlice#sequential_31/lstm_76/transpose:y:04sequential_31/lstm_76/strided_slice_2/stack:output:06sequential_31/lstm_76/strided_slice_2/stack_1:output:06sequential_31/lstm_76/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2'
%sequential_31/lstm_76/strided_slice_2╝
2sequential_31/lstm_76/lstm_cell_76/ones_like/ShapeShape$sequential_31/lstm_76/zeros:output:0*
T0*
_output_shapes
:24
2sequential_31/lstm_76/lstm_cell_76/ones_like/Shapeн
2sequential_31/lstm_76/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?24
2sequential_31/lstm_76/lstm_cell_76/ones_like/ConstР
,sequential_31/lstm_76/lstm_cell_76/ones_likeFill;sequential_31/lstm_76/lstm_cell_76/ones_like/Shape:output:0;sequential_31/lstm_76/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/ones_likeк
2sequential_31/lstm_76/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_31/lstm_76/lstm_cell_76/split/split_dimЇ
7sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOpReadVariableOp@sequential_31_lstm_76_lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOp│
(sequential_31/lstm_76/lstm_cell_76/splitSplit;sequential_31/lstm_76/lstm_cell_76/split/split_dim:output:0?sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2*
(sequential_31/lstm_76/lstm_cell_76/splitї
)sequential_31/lstm_76/lstm_cell_76/MatMulMatMul.sequential_31/lstm_76/strided_slice_2:output:01sequential_31/lstm_76/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2+
)sequential_31/lstm_76/lstm_cell_76/MatMul∙
+sequential_31/lstm_76/lstm_cell_76/MatMul_1MatMul.sequential_31/lstm_76/strided_slice_2:output:01sequential_31/lstm_76/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_1∙
+sequential_31/lstm_76/lstm_cell_76/MatMul_2MatMul.sequential_31/lstm_76/strided_slice_2:output:01sequential_31/lstm_76/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_2∙
+sequential_31/lstm_76/lstm_cell_76/MatMul_3MatMul.sequential_31/lstm_76/strided_slice_2:output:01sequential_31/lstm_76/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_3о
4sequential_31/lstm_76/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_31/lstm_76/lstm_cell_76/split_1/split_dimЎ
9sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOpReadVariableOpBsequential_31_lstm_76_lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOpл
*sequential_31/lstm_76/lstm_cell_76/split_1Split=sequential_31/lstm_76/lstm_cell_76/split_1/split_dim:output:0Asequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2,
*sequential_31/lstm_76/lstm_cell_76/split_1 
*sequential_31/lstm_76/lstm_cell_76/BiasAddBiasAdd3sequential_31/lstm_76/lstm_cell_76/MatMul:product:03sequential_31/lstm_76/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2,
*sequential_31/lstm_76/lstm_cell_76/BiasAddЕ
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_1BiasAdd5sequential_31/lstm_76/lstm_cell_76/MatMul_1:product:03sequential_31/lstm_76/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_1Е
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_2BiasAdd5sequential_31/lstm_76/lstm_cell_76/MatMul_2:product:03sequential_31/lstm_76/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_2Е
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_3BiasAdd5sequential_31/lstm_76/lstm_cell_76/MatMul_3:product:03sequential_31/lstm_76/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/BiasAdd_3ц
&sequential_31/lstm_76/lstm_cell_76/mulMul$sequential_31/lstm_76/zeros:output:05sequential_31/lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2(
&sequential_31/lstm_76/lstm_cell_76/mulъ
(sequential_31/lstm_76/lstm_cell_76/mul_1Mul$sequential_31/lstm_76/zeros:output:05sequential_31/lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_1ъ
(sequential_31/lstm_76/lstm_cell_76/mul_2Mul$sequential_31/lstm_76/zeros:output:05sequential_31/lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_2ъ
(sequential_31/lstm_76/lstm_cell_76/mul_3Mul$sequential_31/lstm_76/zeros:output:05sequential_31/lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_3т
1sequential_31/lstm_76/lstm_cell_76/ReadVariableOpReadVariableOp:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype023
1sequential_31/lstm_76/lstm_cell_76/ReadVariableOp┴
6sequential_31/lstm_76/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_31/lstm_76/lstm_cell_76/strided_slice/stack┼
8sequential_31/lstm_76/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_31/lstm_76/lstm_cell_76/strided_slice/stack_1┼
8sequential_31/lstm_76/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_31/lstm_76/lstm_cell_76/strided_slice/stack_2╬
0sequential_31/lstm_76/lstm_cell_76/strided_sliceStridedSlice9sequential_31/lstm_76/lstm_cell_76/ReadVariableOp:value:0?sequential_31/lstm_76/lstm_cell_76/strided_slice/stack:output:0Asequential_31/lstm_76/lstm_cell_76/strided_slice/stack_1:output:0Asequential_31/lstm_76/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_31/lstm_76/lstm_cell_76/strided_slice¤
+sequential_31/lstm_76/lstm_cell_76/MatMul_4MatMul*sequential_31/lstm_76/lstm_cell_76/mul:z:09sequential_31/lstm_76/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_4ў
&sequential_31/lstm_76/lstm_cell_76/addAddV23sequential_31/lstm_76/lstm_cell_76/BiasAdd:output:05sequential_31/lstm_76/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2(
&sequential_31/lstm_76/lstm_cell_76/add┴
*sequential_31/lstm_76/lstm_cell_76/SigmoidSigmoid*sequential_31/lstm_76/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2,
*sequential_31/lstm_76/lstm_cell_76/Sigmoidц
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_1ReadVariableOp:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype025
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_1┼
8sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_1╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_2┌
2sequential_31/lstm_76/lstm_cell_76/strided_slice_1StridedSlice;sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_1:value:0Asequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_1:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_31/lstm_76/lstm_cell_76/strided_slice_1Б
+sequential_31/lstm_76/lstm_cell_76/MatMul_5MatMul,sequential_31/lstm_76/lstm_cell_76/mul_1:z:0;sequential_31/lstm_76/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_5¤
(sequential_31/lstm_76/lstm_cell_76/add_1AddV25sequential_31/lstm_76/lstm_cell_76/BiasAdd_1:output:05sequential_31/lstm_76/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/add_1╟
,sequential_31/lstm_76/lstm_cell_76/Sigmoid_1Sigmoid,sequential_31/lstm_76/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/Sigmoid_1ч
(sequential_31/lstm_76/lstm_cell_76/mul_4Mul0sequential_31/lstm_76/lstm_cell_76/Sigmoid_1:y:0&sequential_31/lstm_76/zeros_1:output:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_4ц
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_2ReadVariableOp:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype025
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_2┼
8sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_1╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_2┌
2sequential_31/lstm_76/lstm_cell_76/strided_slice_2StridedSlice;sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_2:value:0Asequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_1:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_31/lstm_76/lstm_cell_76/strided_slice_2Б
+sequential_31/lstm_76/lstm_cell_76/MatMul_6MatMul,sequential_31/lstm_76/lstm_cell_76/mul_2:z:0;sequential_31/lstm_76/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_6¤
(sequential_31/lstm_76/lstm_cell_76/add_2AddV25sequential_31/lstm_76/lstm_cell_76/BiasAdd_2:output:05sequential_31/lstm_76/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/add_2║
'sequential_31/lstm_76/lstm_cell_76/ReluRelu,sequential_31/lstm_76/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2)
'sequential_31/lstm_76/lstm_cell_76/ReluЇ
(sequential_31/lstm_76/lstm_cell_76/mul_5Mul.sequential_31/lstm_76/lstm_cell_76/Sigmoid:y:05sequential_31/lstm_76/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_5ы
(sequential_31/lstm_76/lstm_cell_76/add_3AddV2,sequential_31/lstm_76/lstm_cell_76/mul_4:z:0,sequential_31/lstm_76/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/add_3ц
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_3ReadVariableOp:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype025
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_3┼
8sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2:
8sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_1╔
:sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_2┌
2sequential_31/lstm_76/lstm_cell_76/strided_slice_3StridedSlice;sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_3:value:0Asequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_1:output:0Csequential_31/lstm_76/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_31/lstm_76/lstm_cell_76/strided_slice_3Б
+sequential_31/lstm_76/lstm_cell_76/MatMul_7MatMul,sequential_31/lstm_76/lstm_cell_76/mul_3:z:0;sequential_31/lstm_76/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2-
+sequential_31/lstm_76/lstm_cell_76/MatMul_7¤
(sequential_31/lstm_76/lstm_cell_76/add_4AddV25sequential_31/lstm_76/lstm_cell_76/BiasAdd_3:output:05sequential_31/lstm_76/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/add_4╟
,sequential_31/lstm_76/lstm_cell_76/Sigmoid_2Sigmoid,sequential_31/lstm_76/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2.
,sequential_31/lstm_76/lstm_cell_76/Sigmoid_2╛
)sequential_31/lstm_76/lstm_cell_76/Relu_1Relu,sequential_31/lstm_76/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2+
)sequential_31/lstm_76/lstm_cell_76/Relu_1°
(sequential_31/lstm_76/lstm_cell_76/mul_6Mul0sequential_31/lstm_76/lstm_cell_76/Sigmoid_2:y:07sequential_31/lstm_76/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2*
(sequential_31/lstm_76/lstm_cell_76/mul_6╗
3sequential_31/lstm_76/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential_31/lstm_76/TensorArrayV2_1/element_shapeР
%sequential_31/lstm_76/TensorArrayV2_1TensorListReserve<sequential_31/lstm_76/TensorArrayV2_1/element_shape:output:0.sequential_31/lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_31/lstm_76/TensorArrayV2_1z
sequential_31/lstm_76/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_31/lstm_76/timeл
.sequential_31/lstm_76/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_31/lstm_76/while/maximum_iterationsЦ
(sequential_31/lstm_76/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_31/lstm_76/while/loop_counter═
sequential_31/lstm_76/whileWhile1sequential_31/lstm_76/while/loop_counter:output:07sequential_31/lstm_76/while/maximum_iterations:output:0#sequential_31/lstm_76/time:output:0.sequential_31/lstm_76/TensorArrayV2_1:handle:0$sequential_31/lstm_76/zeros:output:0&sequential_31/lstm_76/zeros_1:output:0.sequential_31/lstm_76/strided_slice_1:output:0Msequential_31/lstm_76/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_31_lstm_76_lstm_cell_76_split_readvariableop_resourceBsequential_31_lstm_76_lstm_cell_76_split_1_readvariableop_resource:sequential_31_lstm_76_lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_31_lstm_76_while_body_2462350*4
cond,R*
(sequential_31_lstm_76_while_cond_2462349*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
sequential_31/lstm_76/whileс
Fsequential_31/lstm_76/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_31/lstm_76/TensorArrayV2Stack/TensorListStack/element_shape└
8sequential_31/lstm_76/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_31/lstm_76/while:output:3Osequential_31/lstm_76/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02:
8sequential_31/lstm_76/TensorArrayV2Stack/TensorListStackн
+sequential_31/lstm_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_31/lstm_76/strided_slice_3/stackи
-sequential_31/lstm_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_31/lstm_76/strided_slice_3/stack_1и
-sequential_31/lstm_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_31/lstm_76/strided_slice_3/stack_2Ю
%sequential_31/lstm_76/strided_slice_3StridedSliceAsequential_31/lstm_76/TensorArrayV2Stack/TensorListStack:tensor:04sequential_31/lstm_76/strided_slice_3/stack:output:06sequential_31/lstm_76/strided_slice_3/stack_1:output:06sequential_31/lstm_76/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2'
%sequential_31/lstm_76/strided_slice_3е
&sequential_31/lstm_76/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_31/lstm_76/transpose_1/perm¤
!sequential_31/lstm_76/transpose_1	TransposeAsequential_31/lstm_76/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_31/lstm_76/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2#
!sequential_31/lstm_76/transpose_1Т
sequential_31/lstm_76/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_31/lstm_76/runtime╥
,sequential_31/dense_92/MatMul/ReadVariableOpReadVariableOp5sequential_31_dense_92_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02.
,sequential_31/dense_92/MatMul/ReadVariableOpр
sequential_31/dense_92/MatMulMatMul.sequential_31/lstm_76/strided_slice_3:output:04sequential_31/dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
sequential_31/dense_92/MatMul╤
-sequential_31/dense_92/BiasAdd/ReadVariableOpReadVariableOp6sequential_31_dense_92_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_31/dense_92/BiasAdd/ReadVariableOp▌
sequential_31/dense_92/BiasAddBiasAdd'sequential_31/dense_92/MatMul:product:05sequential_31/dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2 
sequential_31/dense_92/BiasAddЭ
sequential_31/dense_92/ReluRelu'sequential_31/dense_92/BiasAdd:output:0*
T0*'
_output_shapes
:          2
sequential_31/dense_92/Relu╥
,sequential_31/dense_93/MatMul/ReadVariableOpReadVariableOp5sequential_31_dense_93_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_31/dense_93/MatMul/ReadVariableOp█
sequential_31/dense_93/MatMulMatMul)sequential_31/dense_92/Relu:activations:04sequential_31/dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_31/dense_93/MatMul╤
-sequential_31/dense_93/BiasAdd/ReadVariableOpReadVariableOp6sequential_31_dense_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_31/dense_93/BiasAdd/ReadVariableOp▌
sequential_31/dense_93/BiasAddBiasAdd'sequential_31/dense_93/MatMul:product:05sequential_31/dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_31/dense_93/BiasAddЧ
sequential_31/reshape_46/ShapeShape'sequential_31/dense_93/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_31/reshape_46/Shapeж
,sequential_31/reshape_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_31/reshape_46/strided_slice/stackк
.sequential_31/reshape_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_31/reshape_46/strided_slice/stack_1к
.sequential_31/reshape_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_31/reshape_46/strided_slice/stack_2°
&sequential_31/reshape_46/strided_sliceStridedSlice'sequential_31/reshape_46/Shape:output:05sequential_31/reshape_46/strided_slice/stack:output:07sequential_31/reshape_46/strided_slice/stack_1:output:07sequential_31/reshape_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_31/reshape_46/strided_sliceЦ
(sequential_31/reshape_46/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_31/reshape_46/Reshape/shape/1Ц
(sequential_31/reshape_46/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_31/reshape_46/Reshape/shape/2Э
&sequential_31/reshape_46/Reshape/shapePack/sequential_31/reshape_46/strided_slice:output:01sequential_31/reshape_46/Reshape/shape/1:output:01sequential_31/reshape_46/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_31/reshape_46/Reshape/shape▀
 sequential_31/reshape_46/ReshapeReshape'sequential_31/dense_93/BiasAdd:output:0/sequential_31/reshape_46/Reshape/shape:output:0*
T0*+
_output_shapes
:         2"
 sequential_31/reshape_46/ReshapeИ
IdentityIdentity)sequential_31/reshape_46/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЎ
NoOpNoOp.^sequential_31/dense_92/BiasAdd/ReadVariableOp-^sequential_31/dense_92/MatMul/ReadVariableOp.^sequential_31/dense_93/BiasAdd/ReadVariableOp-^sequential_31/dense_93/MatMul/ReadVariableOp2^sequential_31/lstm_76/lstm_cell_76/ReadVariableOp4^sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_14^sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_24^sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_38^sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOp:^sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOp^sequential_31/lstm_76/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2^
-sequential_31/dense_92/BiasAdd/ReadVariableOp-sequential_31/dense_92/BiasAdd/ReadVariableOp2\
,sequential_31/dense_92/MatMul/ReadVariableOp,sequential_31/dense_92/MatMul/ReadVariableOp2^
-sequential_31/dense_93/BiasAdd/ReadVariableOp-sequential_31/dense_93/BiasAdd/ReadVariableOp2\
,sequential_31/dense_93/MatMul/ReadVariableOp,sequential_31/dense_93/MatMul/ReadVariableOp2f
1sequential_31/lstm_76/lstm_cell_76/ReadVariableOp1sequential_31/lstm_76/lstm_cell_76/ReadVariableOp2j
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_13sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_12j
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_23sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_22j
3sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_33sequential_31/lstm_76/lstm_cell_76/ReadVariableOp_32r
7sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOp7sequential_31/lstm_76/lstm_cell_76/split/ReadVariableOp2v
9sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOp9sequential_31/lstm_76/lstm_cell_76/split_1/ReadVariableOp2:
sequential_31/lstm_76/whilesequential_31/lstm_76/while:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
иА
е	
while_body_2465531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeК
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3е
while/lstm_cell_76/mulMulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulй
while/lstm_cell_76/mul_1Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1й
while/lstm_cell_76/mul_2Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2й
while/lstm_cell_76/mul_3Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
ф	
и
/__inference_sequential_31_layer_call_fn_2464196

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_31_layer_call_and_return_conditional_losses_24636072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
э
и
E__inference_dense_93_layer_call_and_return_conditional_losses_2463573

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense_93/bias/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdd╛
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity▒
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_93/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ф	
и
/__inference_sequential_31_layer_call_fn_2464215

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_31_layer_call_and_return_conditional_losses_24640342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
в|
Б
#__inference__traced_restore_2466497
file_prefix2
 assignvariableop_dense_92_kernel:  .
 assignvariableop_1_dense_92_bias: 4
"assignvariableop_2_dense_93_kernel: .
 assignvariableop_3_dense_93_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_76_lstm_cell_76_kernel:	АL
9assignvariableop_10_lstm_76_lstm_cell_76_recurrent_kernel:	 А<
-assignvariableop_11_lstm_76_lstm_cell_76_bias:	А#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_92_kernel_m:  6
(assignvariableop_15_adam_dense_92_bias_m: <
*assignvariableop_16_adam_dense_93_kernel_m: 6
(assignvariableop_17_adam_dense_93_bias_m:I
6assignvariableop_18_adam_lstm_76_lstm_cell_76_kernel_m:	АS
@assignvariableop_19_adam_lstm_76_lstm_cell_76_recurrent_kernel_m:	 АC
4assignvariableop_20_adam_lstm_76_lstm_cell_76_bias_m:	А<
*assignvariableop_21_adam_dense_92_kernel_v:  6
(assignvariableop_22_adam_dense_92_bias_v: <
*assignvariableop_23_adam_dense_93_kernel_v: 6
(assignvariableop_24_adam_dense_93_bias_v:I
6assignvariableop_25_adam_lstm_76_lstm_cell_76_kernel_v:	АS
@assignvariableop_26_adam_lstm_76_lstm_cell_76_recurrent_kernel_v:	 АC
4assignvariableop_27_adam_lstm_76_lstm_cell_76_bias_v:	А
identity_29ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╓
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
value╪B╒B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╚
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╜
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_92_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_92_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_93_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_93_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6г
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7в
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8к
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9│
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_76_lstm_cell_76_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┴
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_76_lstm_cell_76_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╡
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_76_lstm_cell_76_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13б
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14▓
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_92_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15░
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_92_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16▓
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_93_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17░
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_93_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╛
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_76_lstm_cell_76_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╚
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_76_lstm_cell_76_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╝
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_76_lstm_cell_76_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21▓
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_92_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22░
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_92_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_93_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24░
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_93_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╛
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_76_lstm_cell_76_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╚
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_76_lstm_cell_76_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_76_lstm_cell_76_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╞
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28f
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_29о
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
Чв
к
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465114
inputs_0=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileF
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3О
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulТ
lstm_cell_76/mul_1Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1Т
lstm_cell_76/mul_2Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2Т
lstm_cell_76/mul_3Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2464981*
condR
while_cond_2464980*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Г
Ў
E__inference_dense_92_layer_call_and_return_conditional_losses_2463551

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ъ	
к
/__inference_sequential_31_layer_call_fn_2463624
input_32
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_31_layer_call_and_return_conditional_losses_24636072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
╩
H
,__inference_reshape_46_layer_call_fn_2466027

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_46_layer_call_and_return_conditional_losses_24635922
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
сб
и
D__inference_lstm_76_layer_call_and_return_conditional_losses_2463532

inputs=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
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
:         2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3О
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulТ
lstm_cell_76/mul_1Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1Т
lstm_cell_76/mul_2Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2Т
lstm_cell_76/mul_3Mulzeros:output:0lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2463399*
condR
while_cond_2463398*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╕
ў
.__inference_lstm_cell_76_layer_call_fn_2466091

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24628562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          2

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
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
┌
╚
while_cond_2465255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2465255___redundant_placeholder05
1while_while_cond_2465255___redundant_placeholder15
1while_while_cond_2465255___redundant_placeholder25
1while_while_cond_2465255___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
┌
╚
while_cond_2462636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2462636___redundant_placeholder05
1while_while_cond_2462636___redundant_placeholder15
1while_while_cond_2462636___redundant_placeholder25
1while_while_cond_2462636___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
Є+
│
J__inference_sequential_31_layer_call_and_return_conditional_losses_2463607

inputs"
lstm_76_2463533:	А
lstm_76_2463535:	А"
lstm_76_2463537:	 А"
dense_92_2463552:  
dense_92_2463554: "
dense_93_2463574: 
dense_93_2463576:
identityИв dense_92/StatefulPartitionedCallв dense_93/StatefulPartitionedCallв/dense_93/bias/Regularizer/Square/ReadVariableOpвlstm_76/StatefulPartitionedCallв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpе
lstm_76/StatefulPartitionedCallStatefulPartitionedCallinputslstm_76_2463533lstm_76_2463535lstm_76_2463537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24635322!
lstm_76/StatefulPartitionedCall╣
 dense_92/StatefulPartitionedCallStatefulPartitionedCall(lstm_76/StatefulPartitionedCall:output:0dense_92_2463552dense_92_2463554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_24635512"
 dense_92/StatefulPartitionedCall║
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2463574dense_93_2463576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_24635732"
 dense_93/StatefulPartitionedCallВ
reshape_46/PartitionedCallPartitionedCall)dense_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_46_layer_call_and_return_conditional_losses_24635922
reshape_46/PartitionedCall╧
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_76_2463533*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mulп
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_93_2463576*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulВ
IdentityIdentity#reshape_46/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall0^dense_93/bias/Regularizer/Square/ReadVariableOp ^lstm_76/StatefulPartitionedCall>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2B
lstm_76/StatefulPartitionedCalllstm_76/StatefulPartitionedCall2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╕	
а
%__inference_signature_wrapper_2464177
input_32
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_24624992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
Р╨
к
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465421
inputs_0=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileF
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like}
lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout/Const│
lstm_cell_76/dropout/MulMullstm_cell_76/ones_like:output:0#lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/MulЗ
lstm_cell_76/dropout/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout/Shape°
1lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2┬▄∙23
1lstm_cell_76/dropout/random_uniform/RandomUniformП
#lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_76/dropout/GreaterEqual/yЄ
!lstm_cell_76/dropout/GreaterEqualGreaterEqual:lstm_cell_76/dropout/random_uniform/RandomUniform:output:0,lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2#
!lstm_cell_76/dropout/GreaterEqualж
lstm_cell_76/dropout/CastCast%lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout/Castо
lstm_cell_76/dropout/Mul_1Mullstm_cell_76/dropout/Mul:z:0lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/Mul_1Б
lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_1/Const╣
lstm_cell_76/dropout_1/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/MulЛ
lstm_cell_76/dropout_1/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_1/Shape¤
3lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2юЕ|25
3lstm_cell_76/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_1/GreaterEqual/y·
#lstm_cell_76/dropout_1/GreaterEqualGreaterEqual<lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_1/GreaterEqualм
lstm_cell_76/dropout_1/CastCast'lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Cast╢
lstm_cell_76/dropout_1/Mul_1Mullstm_cell_76/dropout_1/Mul:z:0lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Mul_1Б
lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_2/Const╣
lstm_cell_76/dropout_2/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/MulЛ
lstm_cell_76/dropout_2/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_2/Shape¤
3lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2И¤G25
3lstm_cell_76/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_2/GreaterEqual/y·
#lstm_cell_76/dropout_2/GreaterEqualGreaterEqual<lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_2/GreaterEqualм
lstm_cell_76/dropout_2/CastCast'lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Cast╢
lstm_cell_76/dropout_2/Mul_1Mullstm_cell_76/dropout_2/Mul:z:0lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Mul_1Б
lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_3/Const╣
lstm_cell_76/dropout_3/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/MulЛ
lstm_cell_76/dropout_3/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_3/Shape■
3lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Д╣╝25
3lstm_cell_76/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_3/GreaterEqual/y·
#lstm_cell_76/dropout_3/GreaterEqualGreaterEqual<lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_3/GreaterEqualм
lstm_cell_76/dropout_3/CastCast'lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Cast╢
lstm_cell_76/dropout_3/Mul_1Mullstm_cell_76/dropout_3/Mul:z:0lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Mul_1~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3Н
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulУ
lstm_cell_76/mul_1Mulzeros:output:0 lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1У
lstm_cell_76/mul_2Mulzeros:output:0 lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2У
lstm_cell_76/mul_3Mulzeros:output:0 lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2465256*
condR
while_cond_2465255*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
иА
е	
while_body_2464981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeК
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3е
while/lstm_cell_76/mulMulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulй
while/lstm_cell_76/mul_1Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1й
while/lstm_cell_76/mul_2Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2й
while/lstm_cell_76/mul_3Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
ў
к
__inference_loss_fn_0_2466051F
8dense_93_bias_regularizer_square_readvariableop_resource:
identityИв/dense_93/bias/Regularizer/Square/ReadVariableOp╫
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_93_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulk
IdentityIdentity!dense_93/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityА
NoOpNoOp0^dense_93/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp
Ес
Ь
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464486

inputsE
2lstm_76_lstm_cell_76_split_readvariableop_resource:	АC
4lstm_76_lstm_cell_76_split_1_readvariableop_resource:	А?
,lstm_76_lstm_cell_76_readvariableop_resource:	 А9
'dense_92_matmul_readvariableop_resource:  6
(dense_92_biasadd_readvariableop_resource: 9
'dense_93_matmul_readvariableop_resource: 6
(dense_93_biasadd_readvariableop_resource:
identityИвdense_92/BiasAdd/ReadVariableOpвdense_92/MatMul/ReadVariableOpвdense_93/BiasAdd/ReadVariableOpвdense_93/MatMul/ReadVariableOpв/dense_93/bias/Regularizer/Square/ReadVariableOpв#lstm_76/lstm_cell_76/ReadVariableOpв%lstm_76/lstm_cell_76/ReadVariableOp_1в%lstm_76/lstm_cell_76/ReadVariableOp_2в%lstm_76/lstm_cell_76/ReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpв)lstm_76/lstm_cell_76/split/ReadVariableOpв+lstm_76/lstm_cell_76/split_1/ReadVariableOpвlstm_76/whileT
lstm_76/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_76/ShapeД
lstm_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice/stackИ
lstm_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_76/strided_slice/stack_1И
lstm_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_76/strided_slice/stack_2Т
lstm_76/strided_sliceStridedSlicelstm_76/Shape:output:0$lstm_76/strided_slice/stack:output:0&lstm_76/strided_slice/stack_1:output:0&lstm_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_76/strided_slicel
lstm_76/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros/mul/yМ
lstm_76/zeros/mulMullstm_76/strided_slice:output:0lstm_76/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros/mulo
lstm_76/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_76/zeros/Less/yЗ
lstm_76/zeros/LessLesslstm_76/zeros/mul:z:0lstm_76/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros/Lessr
lstm_76/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros/packed/1г
lstm_76/zeros/packedPacklstm_76/strided_slice:output:0lstm_76/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_76/zeros/packedo
lstm_76/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/zeros/ConstХ
lstm_76/zerosFilllstm_76/zeros/packed:output:0lstm_76/zeros/Const:output:0*
T0*'
_output_shapes
:          2
lstm_76/zerosp
lstm_76/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros_1/mul/yТ
lstm_76/zeros_1/mulMullstm_76/strided_slice:output:0lstm_76/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros_1/muls
lstm_76/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_76/zeros_1/Less/yП
lstm_76/zeros_1/LessLesslstm_76/zeros_1/mul:z:0lstm_76/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros_1/Lessv
lstm_76/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros_1/packed/1й
lstm_76/zeros_1/packedPacklstm_76/strided_slice:output:0!lstm_76/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_76/zeros_1/packeds
lstm_76/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/zeros_1/ConstЭ
lstm_76/zeros_1Filllstm_76/zeros_1/packed:output:0lstm_76/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_76/zeros_1Е
lstm_76/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_76/transpose/permТ
lstm_76/transpose	Transposeinputslstm_76/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_76/transposeg
lstm_76/Shape_1Shapelstm_76/transpose:y:0*
T0*
_output_shapes
:2
lstm_76/Shape_1И
lstm_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice_1/stackМ
lstm_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_1/stack_1М
lstm_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_1/stack_2Ю
lstm_76/strided_slice_1StridedSlicelstm_76/Shape_1:output:0&lstm_76/strided_slice_1/stack:output:0(lstm_76/strided_slice_1/stack_1:output:0(lstm_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_76/strided_slice_1Х
#lstm_76/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_76/TensorArrayV2/element_shape╥
lstm_76/TensorArrayV2TensorListReserve,lstm_76/TensorArrayV2/element_shape:output:0 lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_76/TensorArrayV2╧
=lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_76/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_76/transpose:y:0Flstm_76/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_76/TensorArrayUnstack/TensorListFromTensorИ
lstm_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice_2/stackМ
lstm_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_2/stack_1М
lstm_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_2/stack_2м
lstm_76/strided_slice_2StridedSlicelstm_76/transpose:y:0&lstm_76/strided_slice_2/stack:output:0(lstm_76/strided_slice_2/stack_1:output:0(lstm_76/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_76/strided_slice_2Т
$lstm_76/lstm_cell_76/ones_like/ShapeShapelstm_76/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_76/lstm_cell_76/ones_like/ShapeС
$lstm_76/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm_76/lstm_cell_76/ones_like/Const╪
lstm_76/lstm_cell_76/ones_likeFill-lstm_76/lstm_cell_76/ones_like/Shape:output:0-lstm_76/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/ones_likeО
$lstm_76/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_76/lstm_cell_76/split/split_dim╩
)lstm_76/lstm_cell_76/split/ReadVariableOpReadVariableOp2lstm_76_lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)lstm_76/lstm_cell_76/split/ReadVariableOp√
lstm_76/lstm_cell_76/splitSplit-lstm_76/lstm_cell_76/split/split_dim:output:01lstm_76/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_76/lstm_cell_76/split╜
lstm_76/lstm_cell_76/MatMulMatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul┴
lstm_76/lstm_cell_76/MatMul_1MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_1┴
lstm_76/lstm_cell_76/MatMul_2MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_2┴
lstm_76/lstm_cell_76/MatMul_3MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_3Т
&lstm_76/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_76/lstm_cell_76/split_1/split_dim╠
+lstm_76/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4lstm_76_lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_76/lstm_cell_76/split_1/ReadVariableOpє
lstm_76/lstm_cell_76/split_1Split/lstm_76/lstm_cell_76/split_1/split_dim:output:03lstm_76/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_76/lstm_cell_76/split_1╟
lstm_76/lstm_cell_76/BiasAddBiasAdd%lstm_76/lstm_cell_76/MatMul:product:0%lstm_76/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/BiasAdd═
lstm_76/lstm_cell_76/BiasAdd_1BiasAdd'lstm_76/lstm_cell_76/MatMul_1:product:0%lstm_76/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_1═
lstm_76/lstm_cell_76/BiasAdd_2BiasAdd'lstm_76/lstm_cell_76/MatMul_2:product:0%lstm_76/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_2═
lstm_76/lstm_cell_76/BiasAdd_3BiasAdd'lstm_76/lstm_cell_76/MatMul_3:product:0%lstm_76/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_3о
lstm_76/lstm_cell_76/mulMullstm_76/zeros:output:0'lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul▓
lstm_76/lstm_cell_76/mul_1Mullstm_76/zeros:output:0'lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_1▓
lstm_76/lstm_cell_76/mul_2Mullstm_76/zeros:output:0'lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_2▓
lstm_76/lstm_cell_76/mul_3Mullstm_76/zeros:output:0'lstm_76/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_3╕
#lstm_76/lstm_cell_76/ReadVariableOpReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_76/lstm_cell_76/ReadVariableOpе
(lstm_76/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_76/lstm_cell_76/strided_slice/stackй
*lstm_76/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_76/lstm_cell_76/strided_slice/stack_1й
*lstm_76/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_76/lstm_cell_76/strided_slice/stack_2·
"lstm_76/lstm_cell_76/strided_sliceStridedSlice+lstm_76/lstm_cell_76/ReadVariableOp:value:01lstm_76/lstm_cell_76/strided_slice/stack:output:03lstm_76/lstm_cell_76/strided_slice/stack_1:output:03lstm_76/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_76/lstm_cell_76/strided_slice┼
lstm_76/lstm_cell_76/MatMul_4MatMullstm_76/lstm_cell_76/mul:z:0+lstm_76/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_4┐
lstm_76/lstm_cell_76/addAddV2%lstm_76/lstm_cell_76/BiasAdd:output:0'lstm_76/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/addЧ
lstm_76/lstm_cell_76/SigmoidSigmoidlstm_76/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Sigmoid╝
%lstm_76/lstm_cell_76/ReadVariableOp_1ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_1й
*lstm_76/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_76/lstm_cell_76/strided_slice_1/stackн
,lstm_76/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_76/lstm_cell_76/strided_slice_1/stack_1н
,lstm_76/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_1/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_1StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_1:value:03lstm_76/lstm_cell_76/strided_slice_1/stack:output:05lstm_76/lstm_cell_76/strided_slice_1/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_1╔
lstm_76/lstm_cell_76/MatMul_5MatMullstm_76/lstm_cell_76/mul_1:z:0-lstm_76/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_5┼
lstm_76/lstm_cell_76/add_1AddV2'lstm_76/lstm_cell_76/BiasAdd_1:output:0'lstm_76/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_1Э
lstm_76/lstm_cell_76/Sigmoid_1Sigmoidlstm_76/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/Sigmoid_1п
lstm_76/lstm_cell_76/mul_4Mul"lstm_76/lstm_cell_76/Sigmoid_1:y:0lstm_76/zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_4╝
%lstm_76/lstm_cell_76/ReadVariableOp_2ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_2й
*lstm_76/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_76/lstm_cell_76/strided_slice_2/stackн
,lstm_76/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_76/lstm_cell_76/strided_slice_2/stack_1н
,lstm_76/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_2/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_2StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_2:value:03lstm_76/lstm_cell_76/strided_slice_2/stack:output:05lstm_76/lstm_cell_76/strided_slice_2/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_2╔
lstm_76/lstm_cell_76/MatMul_6MatMullstm_76/lstm_cell_76/mul_2:z:0-lstm_76/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_6┼
lstm_76/lstm_cell_76/add_2AddV2'lstm_76/lstm_cell_76/BiasAdd_2:output:0'lstm_76/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_2Р
lstm_76/lstm_cell_76/ReluRelulstm_76/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Relu╝
lstm_76/lstm_cell_76/mul_5Mul lstm_76/lstm_cell_76/Sigmoid:y:0'lstm_76/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_5│
lstm_76/lstm_cell_76/add_3AddV2lstm_76/lstm_cell_76/mul_4:z:0lstm_76/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_3╝
%lstm_76/lstm_cell_76/ReadVariableOp_3ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_3й
*lstm_76/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_76/lstm_cell_76/strided_slice_3/stackн
,lstm_76/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_76/lstm_cell_76/strided_slice_3/stack_1н
,lstm_76/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_3/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_3StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_3:value:03lstm_76/lstm_cell_76/strided_slice_3/stack:output:05lstm_76/lstm_cell_76/strided_slice_3/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_3╔
lstm_76/lstm_cell_76/MatMul_7MatMullstm_76/lstm_cell_76/mul_3:z:0-lstm_76/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_7┼
lstm_76/lstm_cell_76/add_4AddV2'lstm_76/lstm_cell_76/BiasAdd_3:output:0'lstm_76/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_4Э
lstm_76/lstm_cell_76/Sigmoid_2Sigmoidlstm_76/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/Sigmoid_2Ф
lstm_76/lstm_cell_76/Relu_1Relulstm_76/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Relu_1└
lstm_76/lstm_cell_76/mul_6Mul"lstm_76/lstm_cell_76/Sigmoid_2:y:0)lstm_76/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_6Я
%lstm_76/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_76/TensorArrayV2_1/element_shape╪
lstm_76/TensorArrayV2_1TensorListReserve.lstm_76/TensorArrayV2_1/element_shape:output:0 lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_76/TensorArrayV2_1^
lstm_76/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/timeП
 lstm_76/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_76/while/maximum_iterationsz
lstm_76/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/while/loop_counter√
lstm_76/whileWhile#lstm_76/while/loop_counter:output:0)lstm_76/while/maximum_iterations:output:0lstm_76/time:output:0 lstm_76/TensorArrayV2_1:handle:0lstm_76/zeros:output:0lstm_76/zeros_1:output:0 lstm_76/strided_slice_1:output:0?lstm_76/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_76_lstm_cell_76_split_readvariableop_resource4lstm_76_lstm_cell_76_split_1_readvariableop_resource,lstm_76_lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_76_while_body_2464325*&
condR
lstm_76_while_cond_2464324*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
lstm_76/while┼
8lstm_76/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2:
8lstm_76/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_76/TensorArrayV2Stack/TensorListStackTensorListStacklstm_76/while:output:3Alstm_76/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02,
*lstm_76/TensorArrayV2Stack/TensorListStackС
lstm_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_76/strided_slice_3/stackМ
lstm_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_76/strided_slice_3/stack_1М
lstm_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_3/stack_2╩
lstm_76/strided_slice_3StridedSlice3lstm_76/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_76/strided_slice_3/stack:output:0(lstm_76/strided_slice_3/stack_1:output:0(lstm_76/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
lstm_76/strided_slice_3Й
lstm_76/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_76/transpose_1/perm┼
lstm_76/transpose_1	Transpose3lstm_76/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_76/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
lstm_76/transpose_1v
lstm_76/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/runtimeи
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_92/MatMul/ReadVariableOpи
dense_92/MatMulMatMul lstm_76/strided_slice_3:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_92/MatMulз
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_92/BiasAdd/ReadVariableOpе
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_92/BiasAdds
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_92/Reluи
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_93/MatMul/ReadVariableOpг
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_93/MatMulз
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_93/BiasAdd/ReadVariableOpе
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_93/BiasAddm
reshape_46/ShapeShapedense_93/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_46/ShapeК
reshape_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_46/strided_slice/stackО
 reshape_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_46/strided_slice/stack_1О
 reshape_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_46/strided_slice/stack_2д
reshape_46/strided_sliceStridedSlicereshape_46/Shape:output:0'reshape_46/strided_slice/stack:output:0)reshape_46/strided_slice/stack_1:output:0)reshape_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_46/strided_slicez
reshape_46/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_46/Reshape/shape/1z
reshape_46/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_46/Reshape/shape/2╫
reshape_46/Reshape/shapePack!reshape_46/strided_slice:output:0#reshape_46/Reshape/shape/1:output:0#reshape_46/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_46/Reshape/shapeз
reshape_46/ReshapeReshapedense_93/BiasAdd:output:0!reshape_46/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_46/ReshapeЄ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_76_lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mul╟
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulz
IdentityIdentityreshape_46/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╬
NoOpNoOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp0^dense_93/bias/Regularizer/Square/ReadVariableOp$^lstm_76/lstm_cell_76/ReadVariableOp&^lstm_76/lstm_cell_76/ReadVariableOp_1&^lstm_76/lstm_cell_76/ReadVariableOp_2&^lstm_76/lstm_cell_76/ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*^lstm_76/lstm_cell_76/split/ReadVariableOp,^lstm_76/lstm_cell_76/split_1/ReadVariableOp^lstm_76/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2J
#lstm_76/lstm_cell_76/ReadVariableOp#lstm_76/lstm_cell_76/ReadVariableOp2N
%lstm_76/lstm_cell_76/ReadVariableOp_1%lstm_76/lstm_cell_76/ReadVariableOp_12N
%lstm_76/lstm_cell_76/ReadVariableOp_2%lstm_76/lstm_cell_76/ReadVariableOp_22N
%lstm_76/lstm_cell_76/ReadVariableOp_3%lstm_76/lstm_cell_76/ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_76/lstm_cell_76/split/ReadVariableOp)lstm_76/lstm_cell_76/split/ReadVariableOp2Z
+lstm_76/lstm_cell_76/split_1/ReadVariableOp+lstm_76/lstm_cell_76/split_1/ReadVariableOp2
lstm_76/whilelstm_76/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
└
╕
)__inference_lstm_76_layer_call_fn_2464838
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24627122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╧R
ъ
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2462623

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpX
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
 *  А?2
ones_like/ConstД
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:          2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02
split/ReadVariableOpз
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:          2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:          2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:          2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:          2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:          2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:          2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:          2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:          2
	BiasAdd_3_
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:          2
mulc
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:          2
mul_1c
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:          2
mul_2c
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:          2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
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
strided_slice/stack_2№
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
:          2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:          2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2И
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
:          2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:          2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2И
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
:          2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:          2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:          2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:          2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2И
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
:          2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:          2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_6▌
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
є
Ч
*__inference_dense_93_layer_call_fn_2466006

inputs
unknown: 
	unknown_0:
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_24635732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┌
╚
while_cond_2465805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2465805___redundant_placeholder05
1while_while_cond_2465805___redundant_placeholder15
1while_while_cond_2465805___redundant_placeholder25
1while_while_cond_2465805___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
ЪR
╔
D__inference_lstm_76_layer_call_and_return_conditional_losses_2463009

inputs'
lstm_cell_76_2462921:	А#
lstm_cell_76_2462923:	А'
lstm_cell_76_2462925:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpв$lstm_cell_76/StatefulPartitionedCallвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2б
$lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_76_2462921lstm_cell_76_2462923lstm_cell_76_2462925*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24628562&
$lstm_cell_76/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_76_2462921lstm_cell_76_2462923lstm_cell_76_2462925*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2462934*
condR
while_cond_2462933*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime╘
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_76_2462921*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity╜
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_76/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_76/StatefulPartitionedCall$lstm_cell_76/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
бХ
Ь
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464821

inputsE
2lstm_76_lstm_cell_76_split_readvariableop_resource:	АC
4lstm_76_lstm_cell_76_split_1_readvariableop_resource:	А?
,lstm_76_lstm_cell_76_readvariableop_resource:	 А9
'dense_92_matmul_readvariableop_resource:  6
(dense_92_biasadd_readvariableop_resource: 9
'dense_93_matmul_readvariableop_resource: 6
(dense_93_biasadd_readvariableop_resource:
identityИвdense_92/BiasAdd/ReadVariableOpвdense_92/MatMul/ReadVariableOpвdense_93/BiasAdd/ReadVariableOpвdense_93/MatMul/ReadVariableOpв/dense_93/bias/Regularizer/Square/ReadVariableOpв#lstm_76/lstm_cell_76/ReadVariableOpв%lstm_76/lstm_cell_76/ReadVariableOp_1в%lstm_76/lstm_cell_76/ReadVariableOp_2в%lstm_76/lstm_cell_76/ReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpв)lstm_76/lstm_cell_76/split/ReadVariableOpв+lstm_76/lstm_cell_76/split_1/ReadVariableOpвlstm_76/whileT
lstm_76/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_76/ShapeД
lstm_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice/stackИ
lstm_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_76/strided_slice/stack_1И
lstm_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_76/strided_slice/stack_2Т
lstm_76/strided_sliceStridedSlicelstm_76/Shape:output:0$lstm_76/strided_slice/stack:output:0&lstm_76/strided_slice/stack_1:output:0&lstm_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_76/strided_slicel
lstm_76/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros/mul/yМ
lstm_76/zeros/mulMullstm_76/strided_slice:output:0lstm_76/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros/mulo
lstm_76/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_76/zeros/Less/yЗ
lstm_76/zeros/LessLesslstm_76/zeros/mul:z:0lstm_76/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros/Lessr
lstm_76/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros/packed/1г
lstm_76/zeros/packedPacklstm_76/strided_slice:output:0lstm_76/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_76/zeros/packedo
lstm_76/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/zeros/ConstХ
lstm_76/zerosFilllstm_76/zeros/packed:output:0lstm_76/zeros/Const:output:0*
T0*'
_output_shapes
:          2
lstm_76/zerosp
lstm_76/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros_1/mul/yТ
lstm_76/zeros_1/mulMullstm_76/strided_slice:output:0lstm_76/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros_1/muls
lstm_76/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_76/zeros_1/Less/yП
lstm_76/zeros_1/LessLesslstm_76/zeros_1/mul:z:0lstm_76/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_76/zeros_1/Lessv
lstm_76/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/zeros_1/packed/1й
lstm_76/zeros_1/packedPacklstm_76/strided_slice:output:0!lstm_76/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_76/zeros_1/packeds
lstm_76/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/zeros_1/ConstЭ
lstm_76/zeros_1Filllstm_76/zeros_1/packed:output:0lstm_76/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_76/zeros_1Е
lstm_76/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_76/transpose/permТ
lstm_76/transpose	Transposeinputslstm_76/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_76/transposeg
lstm_76/Shape_1Shapelstm_76/transpose:y:0*
T0*
_output_shapes
:2
lstm_76/Shape_1И
lstm_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice_1/stackМ
lstm_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_1/stack_1М
lstm_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_1/stack_2Ю
lstm_76/strided_slice_1StridedSlicelstm_76/Shape_1:output:0&lstm_76/strided_slice_1/stack:output:0(lstm_76/strided_slice_1/stack_1:output:0(lstm_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_76/strided_slice_1Х
#lstm_76/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_76/TensorArrayV2/element_shape╥
lstm_76/TensorArrayV2TensorListReserve,lstm_76/TensorArrayV2/element_shape:output:0 lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_76/TensorArrayV2╧
=lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=lstm_76/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_76/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_76/transpose:y:0Flstm_76/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_76/TensorArrayUnstack/TensorListFromTensorИ
lstm_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_76/strided_slice_2/stackМ
lstm_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_2/stack_1М
lstm_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_2/stack_2м
lstm_76/strided_slice_2StridedSlicelstm_76/transpose:y:0&lstm_76/strided_slice_2/stack:output:0(lstm_76/strided_slice_2/stack_1:output:0(lstm_76/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_76/strided_slice_2Т
$lstm_76/lstm_cell_76/ones_like/ShapeShapelstm_76/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_76/lstm_cell_76/ones_like/ShapeС
$lstm_76/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm_76/lstm_cell_76/ones_like/Const╪
lstm_76/lstm_cell_76/ones_likeFill-lstm_76/lstm_cell_76/ones_like/Shape:output:0-lstm_76/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/ones_likeН
"lstm_76/lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2$
"lstm_76/lstm_cell_76/dropout/Const╙
 lstm_76/lstm_cell_76/dropout/MulMul'lstm_76/lstm_cell_76/ones_like:output:0+lstm_76/lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2"
 lstm_76/lstm_cell_76/dropout/MulЯ
"lstm_76/lstm_cell_76/dropout/ShapeShape'lstm_76/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_76/lstm_cell_76/dropout/ShapeР
9lstm_76/lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform+lstm_76/lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╬Ьп2;
9lstm_76/lstm_cell_76/dropout/random_uniform/RandomUniformЯ
+lstm_76/lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+lstm_76/lstm_cell_76/dropout/GreaterEqual/yТ
)lstm_76/lstm_cell_76/dropout/GreaterEqualGreaterEqualBlstm_76/lstm_cell_76/dropout/random_uniform/RandomUniform:output:04lstm_76/lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2+
)lstm_76/lstm_cell_76/dropout/GreaterEqual╛
!lstm_76/lstm_cell_76/dropout/CastCast-lstm_76/lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2#
!lstm_76/lstm_cell_76/dropout/Cast╬
"lstm_76/lstm_cell_76/dropout/Mul_1Mul$lstm_76/lstm_cell_76/dropout/Mul:z:0%lstm_76/lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2$
"lstm_76/lstm_cell_76/dropout/Mul_1С
$lstm_76/lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2&
$lstm_76/lstm_cell_76/dropout_1/Const┘
"lstm_76/lstm_cell_76/dropout_1/MulMul'lstm_76/lstm_cell_76/ones_like:output:0-lstm_76/lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2$
"lstm_76/lstm_cell_76/dropout_1/Mulг
$lstm_76/lstm_cell_76/dropout_1/ShapeShape'lstm_76/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_76/lstm_cell_76/dropout_1/ShapeЦ
;lstm_76/lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_76/lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2Й▒з2=
;lstm_76/lstm_cell_76/dropout_1/random_uniform/RandomUniformг
-lstm_76/lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_76/lstm_cell_76/dropout_1/GreaterEqual/yЪ
+lstm_76/lstm_cell_76/dropout_1/GreaterEqualGreaterEqualDlstm_76/lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:06lstm_76/lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2-
+lstm_76/lstm_cell_76/dropout_1/GreaterEqual─
#lstm_76/lstm_cell_76/dropout_1/CastCast/lstm_76/lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2%
#lstm_76/lstm_cell_76/dropout_1/Cast╓
$lstm_76/lstm_cell_76/dropout_1/Mul_1Mul&lstm_76/lstm_cell_76/dropout_1/Mul:z:0'lstm_76/lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2&
$lstm_76/lstm_cell_76/dropout_1/Mul_1С
$lstm_76/lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2&
$lstm_76/lstm_cell_76/dropout_2/Const┘
"lstm_76/lstm_cell_76/dropout_2/MulMul'lstm_76/lstm_cell_76/ones_like:output:0-lstm_76/lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2$
"lstm_76/lstm_cell_76/dropout_2/Mulг
$lstm_76/lstm_cell_76/dropout_2/ShapeShape'lstm_76/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_76/lstm_cell_76/dropout_2/ShapeЦ
;lstm_76/lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_76/lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2ч№║2=
;lstm_76/lstm_cell_76/dropout_2/random_uniform/RandomUniformг
-lstm_76/lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_76/lstm_cell_76/dropout_2/GreaterEqual/yЪ
+lstm_76/lstm_cell_76/dropout_2/GreaterEqualGreaterEqualDlstm_76/lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:06lstm_76/lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2-
+lstm_76/lstm_cell_76/dropout_2/GreaterEqual─
#lstm_76/lstm_cell_76/dropout_2/CastCast/lstm_76/lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2%
#lstm_76/lstm_cell_76/dropout_2/Cast╓
$lstm_76/lstm_cell_76/dropout_2/Mul_1Mul&lstm_76/lstm_cell_76/dropout_2/Mul:z:0'lstm_76/lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2&
$lstm_76/lstm_cell_76/dropout_2/Mul_1С
$lstm_76/lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2&
$lstm_76/lstm_cell_76/dropout_3/Const┘
"lstm_76/lstm_cell_76/dropout_3/MulMul'lstm_76/lstm_cell_76/ones_like:output:0-lstm_76/lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2$
"lstm_76/lstm_cell_76/dropout_3/Mulг
$lstm_76/lstm_cell_76/dropout_3/ShapeShape'lstm_76/lstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_76/lstm_cell_76/dropout_3/ShapeЦ
;lstm_76/lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_76/lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╗ыП2=
;lstm_76/lstm_cell_76/dropout_3/random_uniform/RandomUniformг
-lstm_76/lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_76/lstm_cell_76/dropout_3/GreaterEqual/yЪ
+lstm_76/lstm_cell_76/dropout_3/GreaterEqualGreaterEqualDlstm_76/lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:06lstm_76/lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2-
+lstm_76/lstm_cell_76/dropout_3/GreaterEqual─
#lstm_76/lstm_cell_76/dropout_3/CastCast/lstm_76/lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2%
#lstm_76/lstm_cell_76/dropout_3/Cast╓
$lstm_76/lstm_cell_76/dropout_3/Mul_1Mul&lstm_76/lstm_cell_76/dropout_3/Mul:z:0'lstm_76/lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2&
$lstm_76/lstm_cell_76/dropout_3/Mul_1О
$lstm_76/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_76/lstm_cell_76/split/split_dim╩
)lstm_76/lstm_cell_76/split/ReadVariableOpReadVariableOp2lstm_76_lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)lstm_76/lstm_cell_76/split/ReadVariableOp√
lstm_76/lstm_cell_76/splitSplit-lstm_76/lstm_cell_76/split/split_dim:output:01lstm_76/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_76/lstm_cell_76/split╜
lstm_76/lstm_cell_76/MatMulMatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul┴
lstm_76/lstm_cell_76/MatMul_1MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_1┴
lstm_76/lstm_cell_76/MatMul_2MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_2┴
lstm_76/lstm_cell_76/MatMul_3MatMul lstm_76/strided_slice_2:output:0#lstm_76/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_3Т
&lstm_76/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_76/lstm_cell_76/split_1/split_dim╠
+lstm_76/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4lstm_76_lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_76/lstm_cell_76/split_1/ReadVariableOpє
lstm_76/lstm_cell_76/split_1Split/lstm_76/lstm_cell_76/split_1/split_dim:output:03lstm_76/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_76/lstm_cell_76/split_1╟
lstm_76/lstm_cell_76/BiasAddBiasAdd%lstm_76/lstm_cell_76/MatMul:product:0%lstm_76/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/BiasAdd═
lstm_76/lstm_cell_76/BiasAdd_1BiasAdd'lstm_76/lstm_cell_76/MatMul_1:product:0%lstm_76/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_1═
lstm_76/lstm_cell_76/BiasAdd_2BiasAdd'lstm_76/lstm_cell_76/MatMul_2:product:0%lstm_76/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_2═
lstm_76/lstm_cell_76/BiasAdd_3BiasAdd'lstm_76/lstm_cell_76/MatMul_3:product:0%lstm_76/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/BiasAdd_3н
lstm_76/lstm_cell_76/mulMullstm_76/zeros:output:0&lstm_76/lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul│
lstm_76/lstm_cell_76/mul_1Mullstm_76/zeros:output:0(lstm_76/lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_1│
lstm_76/lstm_cell_76/mul_2Mullstm_76/zeros:output:0(lstm_76/lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_2│
lstm_76/lstm_cell_76/mul_3Mullstm_76/zeros:output:0(lstm_76/lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_3╕
#lstm_76/lstm_cell_76/ReadVariableOpReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_76/lstm_cell_76/ReadVariableOpе
(lstm_76/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_76/lstm_cell_76/strided_slice/stackй
*lstm_76/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_76/lstm_cell_76/strided_slice/stack_1й
*lstm_76/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_76/lstm_cell_76/strided_slice/stack_2·
"lstm_76/lstm_cell_76/strided_sliceStridedSlice+lstm_76/lstm_cell_76/ReadVariableOp:value:01lstm_76/lstm_cell_76/strided_slice/stack:output:03lstm_76/lstm_cell_76/strided_slice/stack_1:output:03lstm_76/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_76/lstm_cell_76/strided_slice┼
lstm_76/lstm_cell_76/MatMul_4MatMullstm_76/lstm_cell_76/mul:z:0+lstm_76/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_4┐
lstm_76/lstm_cell_76/addAddV2%lstm_76/lstm_cell_76/BiasAdd:output:0'lstm_76/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/addЧ
lstm_76/lstm_cell_76/SigmoidSigmoidlstm_76/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Sigmoid╝
%lstm_76/lstm_cell_76/ReadVariableOp_1ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_1й
*lstm_76/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_76/lstm_cell_76/strided_slice_1/stackн
,lstm_76/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_76/lstm_cell_76/strided_slice_1/stack_1н
,lstm_76/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_1/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_1StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_1:value:03lstm_76/lstm_cell_76/strided_slice_1/stack:output:05lstm_76/lstm_cell_76/strided_slice_1/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_1╔
lstm_76/lstm_cell_76/MatMul_5MatMullstm_76/lstm_cell_76/mul_1:z:0-lstm_76/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_5┼
lstm_76/lstm_cell_76/add_1AddV2'lstm_76/lstm_cell_76/BiasAdd_1:output:0'lstm_76/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_1Э
lstm_76/lstm_cell_76/Sigmoid_1Sigmoidlstm_76/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/Sigmoid_1п
lstm_76/lstm_cell_76/mul_4Mul"lstm_76/lstm_cell_76/Sigmoid_1:y:0lstm_76/zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_4╝
%lstm_76/lstm_cell_76/ReadVariableOp_2ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_2й
*lstm_76/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_76/lstm_cell_76/strided_slice_2/stackн
,lstm_76/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_76/lstm_cell_76/strided_slice_2/stack_1н
,lstm_76/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_2/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_2StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_2:value:03lstm_76/lstm_cell_76/strided_slice_2/stack:output:05lstm_76/lstm_cell_76/strided_slice_2/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_2╔
lstm_76/lstm_cell_76/MatMul_6MatMullstm_76/lstm_cell_76/mul_2:z:0-lstm_76/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_6┼
lstm_76/lstm_cell_76/add_2AddV2'lstm_76/lstm_cell_76/BiasAdd_2:output:0'lstm_76/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_2Р
lstm_76/lstm_cell_76/ReluRelulstm_76/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Relu╝
lstm_76/lstm_cell_76/mul_5Mul lstm_76/lstm_cell_76/Sigmoid:y:0'lstm_76/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_5│
lstm_76/lstm_cell_76/add_3AddV2lstm_76/lstm_cell_76/mul_4:z:0lstm_76/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_3╝
%lstm_76/lstm_cell_76/ReadVariableOp_3ReadVariableOp,lstm_76_lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_76/lstm_cell_76/ReadVariableOp_3й
*lstm_76/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_76/lstm_cell_76/strided_slice_3/stackн
,lstm_76/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_76/lstm_cell_76/strided_slice_3/stack_1н
,lstm_76/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_76/lstm_cell_76/strided_slice_3/stack_2Ж
$lstm_76/lstm_cell_76/strided_slice_3StridedSlice-lstm_76/lstm_cell_76/ReadVariableOp_3:value:03lstm_76/lstm_cell_76/strided_slice_3/stack:output:05lstm_76/lstm_cell_76/strided_slice_3/stack_1:output:05lstm_76/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_76/lstm_cell_76/strided_slice_3╔
lstm_76/lstm_cell_76/MatMul_7MatMullstm_76/lstm_cell_76/mul_3:z:0-lstm_76/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/MatMul_7┼
lstm_76/lstm_cell_76/add_4AddV2'lstm_76/lstm_cell_76/BiasAdd_3:output:0'lstm_76/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/add_4Э
lstm_76/lstm_cell_76/Sigmoid_2Sigmoidlstm_76/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2 
lstm_76/lstm_cell_76/Sigmoid_2Ф
lstm_76/lstm_cell_76/Relu_1Relulstm_76/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/Relu_1└
lstm_76/lstm_cell_76/mul_6Mul"lstm_76/lstm_cell_76/Sigmoid_2:y:0)lstm_76/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_76/lstm_cell_76/mul_6Я
%lstm_76/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_76/TensorArrayV2_1/element_shape╪
lstm_76/TensorArrayV2_1TensorListReserve.lstm_76/TensorArrayV2_1/element_shape:output:0 lstm_76/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_76/TensorArrayV2_1^
lstm_76/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/timeП
 lstm_76/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_76/while/maximum_iterationsz
lstm_76/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_76/while/loop_counter√
lstm_76/whileWhile#lstm_76/while/loop_counter:output:0)lstm_76/while/maximum_iterations:output:0lstm_76/time:output:0 lstm_76/TensorArrayV2_1:handle:0lstm_76/zeros:output:0lstm_76/zeros_1:output:0 lstm_76/strided_slice_1:output:0?lstm_76/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_76_lstm_cell_76_split_readvariableop_resource4lstm_76_lstm_cell_76_split_1_readvariableop_resource,lstm_76_lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_76_while_body_2464628*&
condR
lstm_76_while_cond_2464627*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
lstm_76/while┼
8lstm_76/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2:
8lstm_76/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_76/TensorArrayV2Stack/TensorListStackTensorListStacklstm_76/while:output:3Alstm_76/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02,
*lstm_76/TensorArrayV2Stack/TensorListStackС
lstm_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_76/strided_slice_3/stackМ
lstm_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_76/strided_slice_3/stack_1М
lstm_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_76/strided_slice_3/stack_2╩
lstm_76/strided_slice_3StridedSlice3lstm_76/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_76/strided_slice_3/stack:output:0(lstm_76/strided_slice_3/stack_1:output:0(lstm_76/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
lstm_76/strided_slice_3Й
lstm_76/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_76/transpose_1/perm┼
lstm_76/transpose_1	Transpose3lstm_76/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_76/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
lstm_76/transpose_1v
lstm_76/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_76/runtimeи
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_92/MatMul/ReadVariableOpи
dense_92/MatMulMatMul lstm_76/strided_slice_3:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_92/MatMulз
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_92/BiasAdd/ReadVariableOpе
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_92/BiasAdds
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_92/Reluи
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_93/MatMul/ReadVariableOpг
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_93/MatMulз
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_93/BiasAdd/ReadVariableOpе
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_93/BiasAddm
reshape_46/ShapeShapedense_93/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_46/ShapeК
reshape_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_46/strided_slice/stackО
 reshape_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_46/strided_slice/stack_1О
 reshape_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_46/strided_slice/stack_2д
reshape_46/strided_sliceStridedSlicereshape_46/Shape:output:0'reshape_46/strided_slice/stack:output:0)reshape_46/strided_slice/stack_1:output:0)reshape_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_46/strided_slicez
reshape_46/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_46/Reshape/shape/1z
reshape_46/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_46/Reshape/shape/2╫
reshape_46/Reshape/shapePack!reshape_46/strided_slice:output:0#reshape_46/Reshape/shape/1:output:0#reshape_46/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_46/Reshape/shapeз
reshape_46/ReshapeReshapedense_93/BiasAdd:output:0!reshape_46/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_46/ReshapeЄ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_76_lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mul╟
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulz
IdentityIdentityreshape_46/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╬
NoOpNoOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp0^dense_93/bias/Regularizer/Square/ReadVariableOp$^lstm_76/lstm_cell_76/ReadVariableOp&^lstm_76/lstm_cell_76/ReadVariableOp_1&^lstm_76/lstm_cell_76/ReadVariableOp_2&^lstm_76/lstm_cell_76/ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*^lstm_76/lstm_cell_76/split/ReadVariableOp,^lstm_76/lstm_cell_76/split_1/ReadVariableOp^lstm_76/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2J
#lstm_76/lstm_cell_76/ReadVariableOp#lstm_76/lstm_cell_76/ReadVariableOp2N
%lstm_76/lstm_cell_76/ReadVariableOp_1%lstm_76/lstm_cell_76/ReadVariableOp_12N
%lstm_76/lstm_cell_76/ReadVariableOp_2%lstm_76/lstm_cell_76/ReadVariableOp_22N
%lstm_76/lstm_cell_76/ReadVariableOp_3%lstm_76/lstm_cell_76/ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_76/lstm_cell_76/split/ReadVariableOp)lstm_76/lstm_cell_76/split/ReadVariableOp2Z
+lstm_76/lstm_cell_76/split_1/ReadVariableOp+lstm_76/lstm_cell_76/split_1/ReadVariableOp2
lstm_76/whilelstm_76/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┌
╚
while_cond_2462933
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2462933___redundant_placeholder05
1while_while_cond_2462933___redundant_placeholder15
1while_while_cond_2462933___redundant_placeholder25
1while_while_cond_2462933___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
┌╧
и
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465971

inputs=
*lstm_cell_76_split_readvariableop_resource:	А;
,lstm_cell_76_split_1_readvariableop_resource:	А7
$lstm_cell_76_readvariableop_resource:	 А
identityИв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвlstm_cell_76/ReadVariableOpвlstm_cell_76/ReadVariableOp_1вlstm_cell_76/ReadVariableOp_2вlstm_cell_76/ReadVariableOp_3в!lstm_cell_76/split/ReadVariableOpв#lstm_cell_76/split_1/ReadVariableOpвwhileD
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
zeros/packed/1Г
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
:          2
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
zeros_1/packed/1Й
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
:          2	
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
:         2
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
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2z
lstm_cell_76/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_76/ones_like/ShapeБ
lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_76/ones_like/Const╕
lstm_cell_76/ones_likeFill%lstm_cell_76/ones_like/Shape:output:0%lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ones_like}
lstm_cell_76/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout/Const│
lstm_cell_76/dropout/MulMullstm_cell_76/ones_like:output:0#lstm_cell_76/dropout/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/MulЗ
lstm_cell_76/dropout/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout/Shape°
1lstm_cell_76/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_76/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2мЯш23
1lstm_cell_76/dropout/random_uniform/RandomUniformП
#lstm_cell_76/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_76/dropout/GreaterEqual/yЄ
!lstm_cell_76/dropout/GreaterEqualGreaterEqual:lstm_cell_76/dropout/random_uniform/RandomUniform:output:0,lstm_cell_76/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2#
!lstm_cell_76/dropout/GreaterEqualж
lstm_cell_76/dropout/CastCast%lstm_cell_76/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout/Castо
lstm_cell_76/dropout/Mul_1Mullstm_cell_76/dropout/Mul:z:0lstm_cell_76/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout/Mul_1Б
lstm_cell_76/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_1/Const╣
lstm_cell_76/dropout_1/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/MulЛ
lstm_cell_76/dropout_1/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_1/Shape¤
3lstm_cell_76/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_1/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2нВY25
3lstm_cell_76/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_1/GreaterEqual/y·
#lstm_cell_76/dropout_1/GreaterEqualGreaterEqual<lstm_cell_76/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_1/GreaterEqualм
lstm_cell_76/dropout_1/CastCast'lstm_cell_76/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Cast╢
lstm_cell_76/dropout_1/Mul_1Mullstm_cell_76/dropout_1/Mul:z:0lstm_cell_76/dropout_1/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_1/Mul_1Б
lstm_cell_76/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_2/Const╣
lstm_cell_76/dropout_2/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_2/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/MulЛ
lstm_cell_76/dropout_2/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_2/Shape■
3lstm_cell_76/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_2/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2ЩЕС25
3lstm_cell_76/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_2/GreaterEqual/y·
#lstm_cell_76/dropout_2/GreaterEqualGreaterEqual<lstm_cell_76/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_2/GreaterEqualм
lstm_cell_76/dropout_2/CastCast'lstm_cell_76/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Cast╢
lstm_cell_76/dropout_2/Mul_1Mullstm_cell_76/dropout_2/Mul:z:0lstm_cell_76/dropout_2/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_2/Mul_1Б
lstm_cell_76/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
lstm_cell_76/dropout_3/Const╣
lstm_cell_76/dropout_3/MulMullstm_cell_76/ones_like:output:0%lstm_cell_76/dropout_3/Const:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/MulЛ
lstm_cell_76/dropout_3/ShapeShapelstm_cell_76/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_76/dropout_3/Shape¤
3lstm_cell_76/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_76/dropout_3/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seed╥	*
seed2╪┬a25
3lstm_cell_76/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_76/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_76/dropout_3/GreaterEqual/y·
#lstm_cell_76/dropout_3/GreaterEqualGreaterEqual<lstm_cell_76/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_76/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2%
#lstm_cell_76/dropout_3/GreaterEqualм
lstm_cell_76/dropout_3/CastCast'lstm_cell_76/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Cast╢
lstm_cell_76/dropout_3/Mul_1Mullstm_cell_76/dropout_3/Mul:z:0lstm_cell_76/dropout_3/Cast:y:0*
T0*'
_output_shapes
:          2
lstm_cell_76/dropout_3/Mul_1~
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_76/split/split_dim▓
!lstm_cell_76/split/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_76/split/ReadVariableOp█
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0)lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_76/splitЭ
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMulб
lstm_cell_76/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_1б
lstm_cell_76/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_2б
lstm_cell_76/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_3В
lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_76/split_1/split_dim┤
#lstm_cell_76/split_1/ReadVariableOpReadVariableOp,lstm_cell_76_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_76/split_1/ReadVariableOp╙
lstm_cell_76/split_1Split'lstm_cell_76/split_1/split_dim:output:0+lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_76/split_1з
lstm_cell_76/BiasAddBiasAddlstm_cell_76/MatMul:product:0lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAddн
lstm_cell_76/BiasAdd_1BiasAddlstm_cell_76/MatMul_1:product:0lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_1н
lstm_cell_76/BiasAdd_2BiasAddlstm_cell_76/MatMul_2:product:0lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_2н
lstm_cell_76/BiasAdd_3BiasAddlstm_cell_76/MatMul_3:product:0lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
lstm_cell_76/BiasAdd_3Н
lstm_cell_76/mulMulzeros:output:0lstm_cell_76/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mulУ
lstm_cell_76/mul_1Mulzeros:output:0 lstm_cell_76/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_1У
lstm_cell_76/mul_2Mulzeros:output:0 lstm_cell_76/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_2У
lstm_cell_76/mul_3Mulzeros:output:0 lstm_cell_76/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_3а
lstm_cell_76/ReadVariableOpReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOpХ
 lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_76/strided_slice/stackЩ
"lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice/stack_1Щ
"lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_76/strided_slice/stack_2╩
lstm_cell_76/strided_sliceStridedSlice#lstm_cell_76/ReadVariableOp:value:0)lstm_cell_76/strided_slice/stack:output:0+lstm_cell_76/strided_slice/stack_1:output:0+lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_sliceе
lstm_cell_76/MatMul_4MatMullstm_cell_76/mul:z:0#lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_4Я
lstm_cell_76/addAddV2lstm_cell_76/BiasAdd:output:0lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add
lstm_cell_76/SigmoidSigmoidlstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoidд
lstm_cell_76/ReadVariableOp_1ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_1Щ
"lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_76/strided_slice_1/stackЭ
$lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_76/strided_slice_1/stack_1Э
$lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_1/stack_2╓
lstm_cell_76/strided_slice_1StridedSlice%lstm_cell_76/ReadVariableOp_1:value:0+lstm_cell_76/strided_slice_1/stack:output:0-lstm_cell_76/strided_slice_1/stack_1:output:0-lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_1й
lstm_cell_76/MatMul_5MatMullstm_cell_76/mul_1:z:0%lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_5е
lstm_cell_76/add_1AddV2lstm_cell_76/BiasAdd_1:output:0lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_1Е
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_1П
lstm_cell_76/mul_4Mullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_4д
lstm_cell_76/ReadVariableOp_2ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_2Щ
"lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_76/strided_slice_2/stackЭ
$lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_76/strided_slice_2/stack_1Э
$lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_2/stack_2╓
lstm_cell_76/strided_slice_2StridedSlice%lstm_cell_76/ReadVariableOp_2:value:0+lstm_cell_76/strided_slice_2/stack:output:0-lstm_cell_76/strided_slice_2/stack_1:output:0-lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_2й
lstm_cell_76/MatMul_6MatMullstm_cell_76/mul_2:z:0%lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_6е
lstm_cell_76/add_2AddV2lstm_cell_76/BiasAdd_2:output:0lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_2x
lstm_cell_76/ReluRelulstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/ReluЬ
lstm_cell_76/mul_5Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_5У
lstm_cell_76/add_3AddV2lstm_cell_76/mul_4:z:0lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_3д
lstm_cell_76/ReadVariableOp_3ReadVariableOp$lstm_cell_76_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_76/ReadVariableOp_3Щ
"lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_76/strided_slice_3/stackЭ
$lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_76/strided_slice_3/stack_1Э
$lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_76/strided_slice_3/stack_2╓
lstm_cell_76/strided_slice_3StridedSlice%lstm_cell_76/ReadVariableOp_3:value:0+lstm_cell_76/strided_slice_3/stack:output:0-lstm_cell_76/strided_slice_3/stack_1:output:0-lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_76/strided_slice_3й
lstm_cell_76/MatMul_7MatMullstm_cell_76/mul_3:z:0%lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
lstm_cell_76/MatMul_7е
lstm_cell_76/add_4AddV2lstm_cell_76/BiasAdd_3:output:0lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
lstm_cell_76/add_4Е
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Sigmoid_2|
lstm_cell_76/Relu_1Relulstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
lstm_cell_76/Relu_1а
lstm_cell_76/mul_6Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_76/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterГ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_76_split_readvariableop_resource,lstm_cell_76_split_1_readvariableop_resource$lstm_cell_76_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2465806*
condR
while_cond_2465805*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_76_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity▐
NoOpNoOp>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_76/ReadVariableOp^lstm_cell_76/ReadVariableOp_1^lstm_cell_76/ReadVariableOp_2^lstm_cell_76/ReadVariableOp_3"^lstm_cell_76/split/ReadVariableOp$^lstm_cell_76/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_76/ReadVariableOplstm_cell_76/ReadVariableOp2>
lstm_cell_76/ReadVariableOp_1lstm_cell_76/ReadVariableOp_12>
lstm_cell_76/ReadVariableOp_2lstm_cell_76/ReadVariableOp_22>
lstm_cell_76/ReadVariableOp_3lstm_cell_76/ReadVariableOp_32F
!lstm_cell_76/split/ReadVariableOp!lstm_cell_76/split/ReadVariableOp2J
#lstm_cell_76/split_1/ReadVariableOp#lstm_cell_76/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▀R
ь
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466172

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3в=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpZ
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
 *  А?2
ones_like/ConstД
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:          2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02
split/ReadVariableOpз
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:          2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:          2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:          2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:          2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:          2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:          2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:          2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:          2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:          2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:          2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:          2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:          2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
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
strided_slice/stack_2№
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
:          2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:          2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2И
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
:          2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:          2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2И
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
:          2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:          2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:          2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:          2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2И
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
:          2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:          2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_6▌
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
иА
е	
while_body_2463399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_76_split_readvariableop_resource_0:	АC
4while_lstm_cell_76_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_76_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_76_split_readvariableop_resource:	АA
2while_lstm_cell_76_split_1_readvariableop_resource:	А=
*while_lstm_cell_76_readvariableop_resource:	 АИв!while/lstm_cell_76/ReadVariableOpв#while/lstm_cell_76/ReadVariableOp_1в#while/lstm_cell_76/ReadVariableOp_2в#while/lstm_cell_76/ReadVariableOp_3в'while/lstm_cell_76/split/ReadVariableOpв)while/lstm_cell_76/split_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_76/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_76/ones_like/ShapeН
"while/lstm_cell_76/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_76/ones_like/Const╨
while/lstm_cell_76/ones_likeFill+while/lstm_cell_76/ones_like/Shape:output:0+while/lstm_cell_76/ones_like/Const:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/ones_likeК
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_76/split/split_dim╞
'while/lstm_cell_76/split/ReadVariableOpReadVariableOp2while_lstm_cell_76_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_76/split/ReadVariableOpє
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0/while/lstm_cell_76/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_76/split╟
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul╦
while/lstm_cell_76/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_1╦
while/lstm_cell_76/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_2╦
while/lstm_cell_76/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_3О
$while/lstm_cell_76/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_76/split_1/split_dim╚
)while/lstm_cell_76/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_76_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_76/split_1/ReadVariableOpы
while/lstm_cell_76/split_1Split-while/lstm_cell_76/split_1/split_dim:output:01while/lstm_cell_76/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_76/split_1┐
while/lstm_cell_76/BiasAddBiasAdd#while/lstm_cell_76/MatMul:product:0#while/lstm_cell_76/split_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd┼
while/lstm_cell_76/BiasAdd_1BiasAdd%while/lstm_cell_76/MatMul_1:product:0#while/lstm_cell_76/split_1:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_1┼
while/lstm_cell_76/BiasAdd_2BiasAdd%while/lstm_cell_76/MatMul_2:product:0#while/lstm_cell_76/split_1:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_2┼
while/lstm_cell_76/BiasAdd_3BiasAdd%while/lstm_cell_76/MatMul_3:product:0#while/lstm_cell_76/split_1:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/BiasAdd_3е
while/lstm_cell_76/mulMulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mulй
while/lstm_cell_76/mul_1Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_1й
while/lstm_cell_76/mul_2Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_2й
while/lstm_cell_76/mul_3Mulwhile_placeholder_2%while/lstm_cell_76/ones_like:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_3┤
!while/lstm_cell_76/ReadVariableOpReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_76/ReadVariableOpб
&while/lstm_cell_76/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_76/strided_slice/stackе
(while/lstm_cell_76/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice/stack_1е
(while/lstm_cell_76/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_76/strided_slice/stack_2ю
 while/lstm_cell_76/strided_sliceStridedSlice)while/lstm_cell_76/ReadVariableOp:value:0/while/lstm_cell_76/strided_slice/stack:output:01while/lstm_cell_76/strided_slice/stack_1:output:01while/lstm_cell_76/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_76/strided_slice╜
while/lstm_cell_76/MatMul_4MatMulwhile/lstm_cell_76/mul:z:0)while/lstm_cell_76/strided_slice:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_4╖
while/lstm_cell_76/addAddV2#while/lstm_cell_76/BiasAdd:output:0%while/lstm_cell_76/MatMul_4:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/addС
while/lstm_cell_76/SigmoidSigmoidwhile/lstm_cell_76/add:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid╕
#while/lstm_cell_76/ReadVariableOp_1ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_1е
(while/lstm_cell_76/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_76/strided_slice_1/stackй
*while/lstm_cell_76/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_76/strided_slice_1/stack_1й
*while/lstm_cell_76/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_1/stack_2·
"while/lstm_cell_76/strided_slice_1StridedSlice+while/lstm_cell_76/ReadVariableOp_1:value:01while/lstm_cell_76/strided_slice_1/stack:output:03while/lstm_cell_76/strided_slice_1/stack_1:output:03while/lstm_cell_76/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_1┴
while/lstm_cell_76/MatMul_5MatMulwhile/lstm_cell_76/mul_1:z:0+while/lstm_cell_76/strided_slice_1:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_5╜
while/lstm_cell_76/add_1AddV2%while/lstm_cell_76/BiasAdd_1:output:0%while/lstm_cell_76/MatMul_5:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_1Ч
while/lstm_cell_76/Sigmoid_1Sigmoidwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_1д
while/lstm_cell_76/mul_4Mul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_4╕
#while/lstm_cell_76/ReadVariableOp_2ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_2е
(while/lstm_cell_76/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_76/strided_slice_2/stackй
*while/lstm_cell_76/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_76/strided_slice_2/stack_1й
*while/lstm_cell_76/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_2/stack_2·
"while/lstm_cell_76/strided_slice_2StridedSlice+while/lstm_cell_76/ReadVariableOp_2:value:01while/lstm_cell_76/strided_slice_2/stack:output:03while/lstm_cell_76/strided_slice_2/stack_1:output:03while/lstm_cell_76/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_2┴
while/lstm_cell_76/MatMul_6MatMulwhile/lstm_cell_76/mul_2:z:0+while/lstm_cell_76/strided_slice_2:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_6╜
while/lstm_cell_76/add_2AddV2%while/lstm_cell_76/BiasAdd_2:output:0%while/lstm_cell_76/MatMul_6:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_2К
while/lstm_cell_76/ReluReluwhile/lstm_cell_76/add_2:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu┤
while/lstm_cell_76/mul_5Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_5л
while/lstm_cell_76/add_3AddV2while/lstm_cell_76/mul_4:z:0while/lstm_cell_76/mul_5:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_3╕
#while/lstm_cell_76/ReadVariableOp_3ReadVariableOp,while_lstm_cell_76_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_76/ReadVariableOp_3е
(while/lstm_cell_76/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_76/strided_slice_3/stackй
*while/lstm_cell_76/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_76/strided_slice_3/stack_1й
*while/lstm_cell_76/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_76/strided_slice_3/stack_2·
"while/lstm_cell_76/strided_slice_3StridedSlice+while/lstm_cell_76/ReadVariableOp_3:value:01while/lstm_cell_76/strided_slice_3/stack:output:03while/lstm_cell_76/strided_slice_3/stack_1:output:03while/lstm_cell_76/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_76/strided_slice_3┴
while/lstm_cell_76/MatMul_7MatMulwhile/lstm_cell_76/mul_3:z:0+while/lstm_cell_76/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/MatMul_7╜
while/lstm_cell_76/add_4AddV2%while/lstm_cell_76/BiasAdd_3:output:0%while/lstm_cell_76/MatMul_7:product:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/add_4Ч
while/lstm_cell_76/Sigmoid_2Sigmoidwhile/lstm_cell_76/add_4:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Sigmoid_2О
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_3:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/Relu_1╕
while/lstm_cell_76/mul_6Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_76/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_76/mul_6:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_76/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_76/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5╞

while/NoOpNoOp"^while/lstm_cell_76/ReadVariableOp$^while/lstm_cell_76/ReadVariableOp_1$^while/lstm_cell_76/ReadVariableOp_2$^while/lstm_cell_76/ReadVariableOp_3(^while/lstm_cell_76/split/ReadVariableOp*^while/lstm_cell_76/split_1/ReadVariableOp*"
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
*while_lstm_cell_76_readvariableop_resource,while_lstm_cell_76_readvariableop_resource_0"j
2while_lstm_cell_76_split_1_readvariableop_resource4while_lstm_cell_76_split_1_readvariableop_resource_0"f
0while_lstm_cell_76_split_readvariableop_resource2while_lstm_cell_76_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2F
!while/lstm_cell_76/ReadVariableOp!while/lstm_cell_76/ReadVariableOp2J
#while/lstm_cell_76/ReadVariableOp_1#while/lstm_cell_76/ReadVariableOp_12J
#while/lstm_cell_76/ReadVariableOp_2#while/lstm_cell_76/ReadVariableOp_22J
#while/lstm_cell_76/ReadVariableOp_3#while/lstm_cell_76/ReadVariableOp_32R
'while/lstm_cell_76/split/ReadVariableOp'while/lstm_cell_76/split/ReadVariableOp2V
)while/lstm_cell_76/split_1/ReadVariableOp)while/lstm_cell_76/split_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
└
╕
)__inference_lstm_76_layer_call_fn_2464849
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24630092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
щ%
ъ
while_body_2462934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_76_2462958_0:	А+
while_lstm_cell_76_2462960_0:	А/
while_lstm_cell_76_2462962_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_76_2462958:	А)
while_lstm_cell_76_2462960:	А-
while_lstm_cell_76_2462962:	 АИв*while/lstm_cell_76/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
*while/lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_76_2462958_0while_lstm_cell_76_2462960_0while_lstm_cell_76_2462962_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_24628562,
*while/lstm_cell_76/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_76/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_76/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_76/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_76/StatefulPartitionedCall*"
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
while_lstm_cell_76_2462958while_lstm_cell_76_2462958_0":
while_lstm_cell_76_2462960while_lstm_cell_76_2462960_0":
while_lstm_cell_76_2462962while_lstm_cell_76_2462962_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_76/StatefulPartitionedCall*while/lstm_cell_76/StatefulPartitionedCall: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╦

ш
lstm_76_while_cond_2464627,
(lstm_76_while_lstm_76_while_loop_counter2
.lstm_76_while_lstm_76_while_maximum_iterations
lstm_76_while_placeholder
lstm_76_while_placeholder_1
lstm_76_while_placeholder_2
lstm_76_while_placeholder_3.
*lstm_76_while_less_lstm_76_strided_slice_1E
Alstm_76_while_lstm_76_while_cond_2464627___redundant_placeholder0E
Alstm_76_while_lstm_76_while_cond_2464627___redundant_placeholder1E
Alstm_76_while_lstm_76_while_cond_2464627___redundant_placeholder2E
Alstm_76_while_lstm_76_while_cond_2464627___redundant_placeholder3
lstm_76_while_identity
Ш
lstm_76/while/LessLesslstm_76_while_placeholder*lstm_76_while_less_lstm_76_strided_slice_1*
T0*
_output_shapes
: 2
lstm_76/while/Lessu
lstm_76/while/IdentityIdentitylstm_76/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_76/while/Identity"9
lstm_76_while_identitylstm_76/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
и
╢
)__inference_lstm_76_layer_call_fn_2464860

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24635322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
э
и
E__inference_dense_93_layer_call_and_return_conditional_losses_2466022

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense_93/bias/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdd╛
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity▒
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_93/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
К
c
G__inference_reshape_46_layer_call_and_return_conditional_losses_2463592

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
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Є+
│
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464034

inputs"
lstm_76_2464003:	А
lstm_76_2464005:	А"
lstm_76_2464007:	 А"
dense_92_2464010:  
dense_92_2464012: "
dense_93_2464015: 
dense_93_2464017:
identityИв dense_92/StatefulPartitionedCallв dense_93/StatefulPartitionedCallв/dense_93/bias/Regularizer/Square/ReadVariableOpвlstm_76/StatefulPartitionedCallв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpе
lstm_76/StatefulPartitionedCallStatefulPartitionedCallinputslstm_76_2464003lstm_76_2464005lstm_76_2464007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24639702!
lstm_76/StatefulPartitionedCall╣
 dense_92/StatefulPartitionedCallStatefulPartitionedCall(lstm_76/StatefulPartitionedCall:output:0dense_92_2464010dense_92_2464012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_24635512"
 dense_92/StatefulPartitionedCall║
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2464015dense_93_2464017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_24635732"
 dense_93/StatefulPartitionedCallВ
reshape_46/PartitionedCallPartitionedCall)dense_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_46_layer_call_and_return_conditional_losses_24635922
reshape_46/PartitionedCall╧
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_76_2464003*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mulп
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_93_2464017*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulВ
IdentityIdentity#reshape_46/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall0^dense_93/bias/Regularizer/Square/ReadVariableOp ^lstm_76/StatefulPartitionedCall>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2B
lstm_76/StatefulPartitionedCalllstm_76/StatefulPartitionedCall2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
°+
╡
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464138
input_32"
lstm_76_2464107:	А
lstm_76_2464109:	А"
lstm_76_2464111:	 А"
dense_92_2464114:  
dense_92_2464116: "
dense_93_2464119: 
dense_93_2464121:
identityИв dense_92/StatefulPartitionedCallв dense_93/StatefulPartitionedCallв/dense_93/bias/Regularizer/Square/ReadVariableOpвlstm_76/StatefulPartitionedCallв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpз
lstm_76/StatefulPartitionedCallStatefulPartitionedCallinput_32lstm_76_2464107lstm_76_2464109lstm_76_2464111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24639702!
lstm_76/StatefulPartitionedCall╣
 dense_92/StatefulPartitionedCallStatefulPartitionedCall(lstm_76/StatefulPartitionedCall:output:0dense_92_2464114dense_92_2464116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_24635512"
 dense_92/StatefulPartitionedCall║
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2464119dense_93_2464121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_24635732"
 dense_93/StatefulPartitionedCallВ
reshape_46/PartitionedCallPartitionedCall)dense_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_46_layer_call_and_return_conditional_losses_24635922
reshape_46/PartitionedCall╧
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_76_2464107*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mulп
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_93_2464121*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulВ
IdentityIdentity#reshape_46/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall0^dense_93/bias/Regularizer/Square/ReadVariableOp ^lstm_76/StatefulPartitionedCall>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2B
lstm_76/StatefulPartitionedCalllstm_76/StatefulPartitionedCall2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
°+
╡
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464104
input_32"
lstm_76_2464073:	А
lstm_76_2464075:	А"
lstm_76_2464077:	 А"
dense_92_2464080:  
dense_92_2464082: "
dense_93_2464085: 
dense_93_2464087:
identityИв dense_92/StatefulPartitionedCallв dense_93/StatefulPartitionedCallв/dense_93/bias/Regularizer/Square/ReadVariableOpвlstm_76/StatefulPartitionedCallв=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpз
lstm_76/StatefulPartitionedCallStatefulPartitionedCallinput_32lstm_76_2464073lstm_76_2464075lstm_76_2464077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_76_layer_call_and_return_conditional_losses_24635322!
lstm_76/StatefulPartitionedCall╣
 dense_92/StatefulPartitionedCallStatefulPartitionedCall(lstm_76/StatefulPartitionedCall:output:0dense_92_2464080dense_92_2464082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_24635512"
 dense_92/StatefulPartitionedCall║
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2464085dense_93_2464087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_24635732"
 dense_93/StatefulPartitionedCallВ
reshape_46/PartitionedCallPartitionedCall)dense_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_46_layer_call_and_return_conditional_losses_24635922
reshape_46/PartitionedCall╧
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_76_2464073*
_output_shapes
:	А*
dtype02?
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp█
.lstm_76/lstm_cell_76/kernel/Regularizer/SquareSquareElstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_76/lstm_cell_76/kernel/Regularizer/Squareп
-lstm_76/lstm_cell_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_76/lstm_cell_76/kernel/Regularizer/Constю
+lstm_76/lstm_cell_76/kernel/Regularizer/SumSum2lstm_76/lstm_cell_76/kernel/Regularizer/Square:y:06lstm_76/lstm_cell_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/Sumг
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82/
-lstm_76/lstm_cell_76/kernel/Regularizer/mul/xЁ
+lstm_76/lstm_cell_76/kernel/Regularizer/mulMul6lstm_76/lstm_cell_76/kernel/Regularizer/mul/x:output:04lstm_76/lstm_cell_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_76/lstm_cell_76/kernel/Regularizer/mulп
/dense_93/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_93_2464087*
_output_shapes
:*
dtype021
/dense_93/bias/Regularizer/Square/ReadVariableOpм
 dense_93/bias/Regularizer/SquareSquare7dense_93/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_93/bias/Regularizer/SquareМ
dense_93/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_93/bias/Regularizer/Const╢
dense_93/bias/Regularizer/SumSum$dense_93/bias/Regularizer/Square:y:0(dense_93/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/SumЗ
dense_93/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2!
dense_93/bias/Regularizer/mul/x╕
dense_93/bias/Regularizer/mulMul(dense_93/bias/Regularizer/mul/x:output:0&dense_93/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_93/bias/Regularizer/mulВ
IdentityIdentity#reshape_46/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall0^dense_93/bias/Regularizer/Square/ReadVariableOp ^lstm_76/StatefulPartitionedCall>^lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2b
/dense_93/bias/Regularizer/Square/ReadVariableOp/dense_93/bias/Regularizer/Square/ReadVariableOp2B
lstm_76/StatefulPartitionedCalllstm_76/StatefulPartitionedCall2~
=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp=lstm_76/lstm_cell_76/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:         
"
_user_specified_name
input_32
┌
╚
while_cond_2465530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2465530___redundant_placeholder05
1while_while_cond_2465530___redundant_placeholder15
1while_while_cond_2465530___redundant_placeholder25
1while_while_cond_2465530___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╖
serving_defaultг
A
input_325
serving_default_input_32:0         B

reshape_464
StatefulPartitionedCall:0         tensorflow/serving/predict:▌В
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
├
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
╗

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
е
trainable_variables
	variables
regularization_losses
 	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
╤
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
╩

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
╣

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
!:  2dense_92/kernel
: 2dense_92/bias
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
н

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
!: 2dense_93/kernel
:2dense_93/bias
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
н

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
н

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
.:,	А2lstm_76/lstm_cell_76/kernel
8:6	 А2%lstm_76/lstm_cell_76/recurrent_kernel
(:&А2lstm_76/lstm_cell_76/bias
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
н

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
&:$  2Adam/dense_92/kernel/m
 : 2Adam/dense_92/bias/m
&:$ 2Adam/dense_93/kernel/m
 :2Adam/dense_93/bias/m
3:1	А2"Adam/lstm_76/lstm_cell_76/kernel/m
=:;	 А2,Adam/lstm_76/lstm_cell_76/recurrent_kernel/m
-:+А2 Adam/lstm_76/lstm_cell_76/bias/m
&:$  2Adam/dense_92/kernel/v
 : 2Adam/dense_92/bias/v
&:$ 2Adam/dense_93/kernel/v
 :2Adam/dense_93/bias/v
3:1	А2"Adam/lstm_76/lstm_cell_76/kernel/v
=:;	 А2,Adam/lstm_76/lstm_cell_76/recurrent_kernel/v
-:+А2 Adam/lstm_76/lstm_cell_76/bias/v
╬B╦
"__inference__wrapped_model_2462499input_32"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
К2З
/__inference_sequential_31_layer_call_fn_2463624
/__inference_sequential_31_layer_call_fn_2464196
/__inference_sequential_31_layer_call_fn_2464215
/__inference_sequential_31_layer_call_fn_2464070└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464486
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464821
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464104
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464138└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
З2Д
)__inference_lstm_76_layer_call_fn_2464838
)__inference_lstm_76_layer_call_fn_2464849
)__inference_lstm_76_layer_call_fn_2464860
)__inference_lstm_76_layer_call_fn_2464871╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465114
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465421
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465664
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465971╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_dense_92_layer_call_fn_2465980в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_92_layer_call_and_return_conditional_losses_2465991в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_93_layer_call_fn_2466006в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_93_layer_call_and_return_conditional_losses_2466022в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_reshape_46_layer_call_fn_2466027в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_reshape_46_layer_call_and_return_conditional_losses_2466040в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤2▒
__inference_loss_fn_0_2466051П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
═B╩
%__inference_signature_wrapper_2464177input_32"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
д2б
.__inference_lstm_cell_76_layer_call_fn_2466074
.__inference_lstm_cell_76_layer_call_fn_2466091╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466172
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466285╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┤2▒
__inference_loss_fn_1_2466296П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в г
"__inference__wrapped_model_2462499}&('5в2
+в(
&К#
input_32         
к ";к8
6

reshape_46(К%

reshape_46         е
E__inference_dense_92_layer_call_and_return_conditional_losses_2465991\/в,
%в"
 К
inputs          
к "%в"
К
0          
Ъ }
*__inference_dense_92_layer_call_fn_2465980O/в,
%в"
 К
inputs          
к "К          е
E__inference_dense_93_layer_call_and_return_conditional_losses_2466022\/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ }
*__inference_dense_93_layer_call_fn_2466006O/в,
%в"
 К
inputs          
к "К         <
__inference_loss_fn_0_2466051в

в 
к "К <
__inference_loss_fn_1_2466296&в

в 
к "К ┼
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465114}&('OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%в"
К
0          
Ъ ┼
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465421}&('OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%в"
К
0          
Ъ ╡
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465664m&('?в<
5в2
$К!
inputs         

 
p 

 
к "%в"
К
0          
Ъ ╡
D__inference_lstm_76_layer_call_and_return_conditional_losses_2465971m&('?в<
5в2
$К!
inputs         

 
p

 
к "%в"
К
0          
Ъ Э
)__inference_lstm_76_layer_call_fn_2464838p&('OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "К          Э
)__inference_lstm_76_layer_call_fn_2464849p&('OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "К          Н
)__inference_lstm_76_layer_call_fn_2464860`&('?в<
5в2
$К!
inputs         

 
p 

 
к "К          Н
)__inference_lstm_76_layer_call_fn_2464871`&('?в<
5в2
$К!
inputs         

 
p

 
к "К          ╦
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466172¤&('Ав}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ ╦
I__inference_lstm_cell_76_layer_call_and_return_conditional_losses_2466285¤&('Ав}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ а
.__inference_lstm_cell_76_layer_call_fn_2466074э&('Ав}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          а
.__inference_lstm_cell_76_layer_call_fn_2466091э&('Ав}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          з
G__inference_reshape_46_layer_call_and_return_conditional_losses_2466040\/в,
%в"
 К
inputs         
к ")в&
К
0         
Ъ 
,__inference_reshape_46_layer_call_fn_2466027O/в,
%в"
 К
inputs         
к "К         ┴
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464104s&('=в:
3в0
&К#
input_32         
p 

 
к ")в&
К
0         
Ъ ┴
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464138s&('=в:
3в0
&К#
input_32         
p

 
к ")в&
К
0         
Ъ ┐
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464486q&(';в8
1в.
$К!
inputs         
p 

 
к ")в&
К
0         
Ъ ┐
J__inference_sequential_31_layer_call_and_return_conditional_losses_2464821q&(';в8
1в.
$К!
inputs         
p

 
к ")в&
К
0         
Ъ Щ
/__inference_sequential_31_layer_call_fn_2463624f&('=в:
3в0
&К#
input_32         
p 

 
к "К         Щ
/__inference_sequential_31_layer_call_fn_2464070f&('=в:
3в0
&К#
input_32         
p

 
к "К         Ч
/__inference_sequential_31_layer_call_fn_2464196d&(';в8
1в.
$К!
inputs         
p 

 
к "К         Ч
/__inference_sequential_31_layer_call_fn_2464215d&(';в8
1в.
$К!
inputs         
p

 
к "К         │
%__inference_signature_wrapper_2464177Й&('Aв>
в 
7к4
2
input_32&К#
input_32         ";к8
6

reshape_46(К%

reshape_46         