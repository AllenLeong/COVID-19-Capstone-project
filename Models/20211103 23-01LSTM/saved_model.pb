≤µ'
ЋЬ
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
Њ
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
ц
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
Ђ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ЎХ&
|
dense_140/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_140/kernel
u
$dense_140/kernel/Read/ReadVariableOpReadVariableOpdense_140/kernel*
_output_shapes

:  *
dtype0
t
dense_140/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_140/bias
m
"dense_140/bias/Read/ReadVariableOpReadVariableOpdense_140/bias*
_output_shapes
: *
dtype0
|
dense_141/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_141/kernel
u
$dense_141/kernel/Read/ReadVariableOpReadVariableOpdense_141/kernel*
_output_shapes

: *
dtype0
t
dense_141/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_141/bias
m
"dense_141/bias/Read/ReadVariableOpReadVariableOpdense_141/bias*
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
Ч
lstm_116/lstm_cell_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*.
shared_namelstm_116/lstm_cell_116/kernel
Р
1lstm_116/lstm_cell_116/kernel/Read/ReadVariableOpReadVariableOplstm_116/lstm_cell_116/kernel*
_output_shapes
:	А*
dtype0
Ђ
'lstm_116/lstm_cell_116/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*8
shared_name)'lstm_116/lstm_cell_116/recurrent_kernel
§
;lstm_116/lstm_cell_116/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_116/lstm_cell_116/recurrent_kernel*
_output_shapes
:	 А*
dtype0
П
lstm_116/lstm_cell_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namelstm_116/lstm_cell_116/bias
И
/lstm_116/lstm_cell_116/bias/Read/ReadVariableOpReadVariableOplstm_116/lstm_cell_116/bias*
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
К
Adam/dense_140/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_140/kernel/m
Г
+Adam/dense_140/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_140/kernel/m*
_output_shapes

:  *
dtype0
В
Adam/dense_140/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_140/bias/m
{
)Adam/dense_140/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_140/bias/m*
_output_shapes
: *
dtype0
К
Adam/dense_141/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_141/kernel/m
Г
+Adam/dense_141/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_141/kernel/m*
_output_shapes

: *
dtype0
В
Adam/dense_141/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_141/bias/m
{
)Adam/dense_141/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_141/bias/m*
_output_shapes
:*
dtype0
•
$Adam/lstm_116/lstm_cell_116/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*5
shared_name&$Adam/lstm_116/lstm_cell_116/kernel/m
Ю
8Adam/lstm_116/lstm_cell_116/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/lstm_116/lstm_cell_116/kernel/m*
_output_shapes
:	А*
dtype0
є
.Adam/lstm_116/lstm_cell_116/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*?
shared_name0.Adam/lstm_116/lstm_cell_116/recurrent_kernel/m
≤
BAdam/lstm_116/lstm_cell_116/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/lstm_116/lstm_cell_116/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Э
"Adam/lstm_116/lstm_cell_116/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/lstm_116/lstm_cell_116/bias/m
Ц
6Adam/lstm_116/lstm_cell_116/bias/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_116/lstm_cell_116/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_140/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_140/kernel/v
Г
+Adam/dense_140/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_140/kernel/v*
_output_shapes

:  *
dtype0
В
Adam/dense_140/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_140/bias/v
{
)Adam/dense_140/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_140/bias/v*
_output_shapes
: *
dtype0
К
Adam/dense_141/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_141/kernel/v
Г
+Adam/dense_141/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_141/kernel/v*
_output_shapes

: *
dtype0
В
Adam/dense_141/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_141/bias/v
{
)Adam/dense_141/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_141/bias/v*
_output_shapes
:*
dtype0
•
$Adam/lstm_116/lstm_cell_116/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*5
shared_name&$Adam/lstm_116/lstm_cell_116/kernel/v
Ю
8Adam/lstm_116/lstm_cell_116/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/lstm_116/lstm_cell_116/kernel/v*
_output_shapes
:	А*
dtype0
є
.Adam/lstm_116/lstm_cell_116/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*?
shared_name0.Adam/lstm_116/lstm_cell_116/recurrent_kernel/v
≤
BAdam/lstm_116/lstm_cell_116/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/lstm_116/lstm_cell_116/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Э
"Adam/lstm_116/lstm_cell_116/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/lstm_116/lstm_cell_116/bias/v
Ц
6Adam/lstm_116/lstm_cell_116/bias/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_116/lstm_cell_116/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
’,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Р,
valueЖ,BГ, Bь+
у
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
Њ
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
≠

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
є

3layers
trainable_variables
	variables
4metrics
5layer_metrics
6layer_regularization_losses
7non_trainable_variables

8states
regularization_losses
\Z
VARIABLE_VALUEdense_140/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_140/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠

9layers
trainable_variables
	variables
:metrics
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_141/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_141/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠

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
≠

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
ca
VARIABLE_VALUElstm_116/lstm_cell_116/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'lstm_116/lstm_cell_116/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_116/lstm_cell_116/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

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
≠

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
}
VARIABLE_VALUEAdam/dense_140/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_140/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_141/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_141/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE$Adam/lstm_116/lstm_cell_116/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/lstm_116/lstm_cell_116/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_116/lstm_cell_116/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_140/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_140/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_141/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_141/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE$Adam/lstm_116/lstm_cell_116/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/lstm_116/lstm_cell_116/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_116/lstm_cell_116/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Г
serving_default_input_48Placeholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_48lstm_116/lstm_cell_116/kernellstm_116/lstm_cell_116/bias'lstm_116/lstm_cell_116/recurrent_kerneldense_140/kerneldense_140/biasdense_141/kerneldense_141/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_3703727
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_140/kernel/Read/ReadVariableOp"dense_140/bias/Read/ReadVariableOp$dense_141/kernel/Read/ReadVariableOp"dense_141/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1lstm_116/lstm_cell_116/kernel/Read/ReadVariableOp;lstm_116/lstm_cell_116/recurrent_kernel/Read/ReadVariableOp/lstm_116/lstm_cell_116/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_140/kernel/m/Read/ReadVariableOp)Adam/dense_140/bias/m/Read/ReadVariableOp+Adam/dense_141/kernel/m/Read/ReadVariableOp)Adam/dense_141/bias/m/Read/ReadVariableOp8Adam/lstm_116/lstm_cell_116/kernel/m/Read/ReadVariableOpBAdam/lstm_116/lstm_cell_116/recurrent_kernel/m/Read/ReadVariableOp6Adam/lstm_116/lstm_cell_116/bias/m/Read/ReadVariableOp+Adam/dense_140/kernel/v/Read/ReadVariableOp)Adam/dense_140/bias/v/Read/ReadVariableOp+Adam/dense_141/kernel/v/Read/ReadVariableOp)Adam/dense_141/bias/v/Read/ReadVariableOp8Adam/lstm_116/lstm_cell_116/kernel/v/Read/ReadVariableOpBAdam/lstm_116/lstm_cell_116/recurrent_kernel/v/Read/ReadVariableOp6Adam/lstm_116/lstm_cell_116/bias/v/Read/ReadVariableOpConst*)
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
 __inference__traced_save_3705953
г
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_140/kerneldense_140/biasdense_141/kerneldense_141/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_116/lstm_cell_116/kernel'lstm_116/lstm_cell_116/recurrent_kernellstm_116/lstm_cell_116/biastotalcountAdam/dense_140/kernel/mAdam/dense_140/bias/mAdam/dense_141/kernel/mAdam/dense_141/bias/m$Adam/lstm_116/lstm_cell_116/kernel/m.Adam/lstm_116/lstm_cell_116/recurrent_kernel/m"Adam/lstm_116/lstm_cell_116/bias/mAdam/dense_140/kernel/vAdam/dense_140/bias/vAdam/dense_141/kernel/vAdam/dense_141/bias/v$Adam/lstm_116/lstm_cell_116/kernel/v.Adam/lstm_116/lstm_cell_116/recurrent_kernel/v"Adam/lstm_116/lstm_cell_116/bias/v*(
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
#__inference__traced_restore_3706047щФ%
вv
п
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705835

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
:€€€€€€€€€ 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape—
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2†ыЕ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_1/ConstЕ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape„
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2…уг2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_1/GreaterEqual/y∆
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_2/ConstЕ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape÷
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЋИ]2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_2/GreaterEqual/y∆
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_3/ConstЕ
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape÷
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Жпg2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_3/GreaterEqual/y∆
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
:€€€€€€€€€ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice/stack_2ь
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
:€€€€€€€€€ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
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
:€€€€€€€€€ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6б
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_2К
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/1
В&
с
while_body_3702484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_116_3702508_0:	А,
while_lstm_cell_116_3702510_0:	А0
while_lstm_cell_116_3702512_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_116_3702508:	А*
while_lstm_cell_116_3702510:	А.
while_lstm_cell_116_3702512:	 АИҐ+while/lstm_cell_116/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemл
+while/lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_116_3702508_0while_lstm_cell_116_3702510_0while_lstm_cell_116_3702512_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37024062-
+while/lstm_cell_116/StatefulPartitionedCallш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_116/StatefulPartitionedCall:output:0*
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
while/Identity_3•
while/Identity_4Identity4while/lstm_cell_116/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4•
while/Identity_5Identity4while/lstm_cell_116/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5И

while/NoOpNoOp,^while/lstm_cell_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_116_3702508while_lstm_cell_116_3702508_0"<
while_lstm_cell_116_3702510while_lstm_cell_116_3702510_0"<
while_lstm_cell_116_3702512while_lstm_cell_116_3702512_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2Z
+while/lstm_cell_116/StatefulPartitionedCall+while/lstm_cell_116/StatefulPartitionedCall: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Џ,
ј
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703157

inputs#
lstm_116_3703083:	А
lstm_116_3703085:	А#
lstm_116_3703087:	 А#
dense_140_3703102:  
dense_140_3703104: #
dense_141_3703124: 
dense_141_3703126:
identityИҐ!dense_140/StatefulPartitionedCallҐ!dense_141/StatefulPartitionedCallҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ lstm_116/StatefulPartitionedCallҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpЂ
 lstm_116/StatefulPartitionedCallStatefulPartitionedCallinputslstm_116_3703083lstm_116_3703085lstm_116_3703087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37030822"
 lstm_116/StatefulPartitionedCallњ
!dense_140/StatefulPartitionedCallStatefulPartitionedCall)lstm_116/StatefulPartitionedCall:output:0dense_140_3703102dense_140_3703104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_37031012#
!dense_140/StatefulPartitionedCallј
!dense_141/StatefulPartitionedCallStatefulPartitionedCall*dense_140/StatefulPartitionedCall:output:0dense_141_3703124dense_141_3703126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_37031232#
!dense_141/StatefulPartitionedCallГ
reshape_70/PartitionedCallPartitionedCall*dense_141/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_70_layer_call_and_return_conditional_losses_37031422
reshape_70/PartitionedCall‘
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_116_3703083*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul≤
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_141_3703126*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulВ
IdentityIdentity#reshape_70/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЃ
NoOpNoOp"^dense_140/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall1^dense_141/bias/Regularizer/Square/ReadVariableOp!^lstm_116/StatefulPartitionedCall@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2D
 lstm_116/StatefulPartitionedCall lstm_116/StatefulPartitionedCall2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
Ш
+__inference_dense_140_layer_call_fn_3705530

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_37031012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ЛВ
±	
while_body_3702949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeМ
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3®
while/lstm_cell_116/mulMulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mulђ
while/lstm_cell_116/mul_1Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1ђ
while/lstm_cell_116/mul_2Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2ђ
while/lstm_cell_116/mul_3Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
¬
є
*__inference_lstm_116_layer_call_fn_3704388
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37022622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
¬
є
*__inference_lstm_116_layer_call_fn_3704399
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37025592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
 
H
,__inference_reshape_70_layer_call_fn_3705577

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_70_layer_call_and_return_conditional_losses_37031422
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Г
™
F__inference_dense_141_layer_call_and_return_conditional_losses_3703123

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_141/bias/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddј
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity≤
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_141/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Џ
»
while_cond_3705080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3705080___redundant_placeholder05
1while_while_cond_3705080___redundant_placeholder15
1while_while_cond_3705080___redundant_placeholder25
1while_while_cond_3705080___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
√µ
±	
while_body_3703355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeЛ
!while/lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2#
!while/lstm_cell_116/dropout/Constѕ
while/lstm_cell_116/dropout/MulMul&while/lstm_cell_116/ones_like:output:0*while/lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_116/dropout/MulЬ
!while/lstm_cell_116/dropout/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_116/dropout/ShapeН
8while/lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2—Йх2:
8while/lstm_cell_116/dropout/random_uniform/RandomUniformЭ
*while/lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2,
*while/lstm_cell_116/dropout/GreaterEqual/yО
(while/lstm_cell_116/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_116/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_116/dropout/GreaterEqualї
 while/lstm_cell_116/dropout/CastCast,while/lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_116/dropout/Cast 
!while/lstm_cell_116/dropout/Mul_1Mul#while/lstm_cell_116/dropout/Mul:z:0$while/lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout/Mul_1П
#while/lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_1/Const’
!while/lstm_cell_116/dropout_1/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_1/Mul†
#while/lstm_cell_116/dropout_1/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_1/ShapeУ
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2“ћ√2<
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_1/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_1/GreaterEqualЅ
"while/lstm_cell_116/dropout_1/CastCast.while/lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_1/Cast“
#while/lstm_cell_116/dropout_1/Mul_1Mul%while/lstm_cell_116/dropout_1/Mul:z:0&while/lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_1/Mul_1П
#while/lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_2/Const’
!while/lstm_cell_116/dropout_2/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_2/Mul†
#while/lstm_cell_116/dropout_2/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_2/ShapeУ
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Ѓх≠2<
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_2/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_2/GreaterEqualЅ
"while/lstm_cell_116/dropout_2/CastCast.while/lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_2/Cast“
#while/lstm_cell_116/dropout_2/Mul_1Mul%while/lstm_cell_116/dropout_2/Mul:z:0&while/lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_2/Mul_1П
#while/lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_3/Const’
!while/lstm_cell_116/dropout_3/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_3/Mul†
#while/lstm_cell_116/dropout_3/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_3/ShapeУ
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2к÷л2<
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_3/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_3/GreaterEqualЅ
"while/lstm_cell_116/dropout_3/CastCast.while/lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_3/Cast“
#while/lstm_cell_116/dropout_3/Mul_1Mul%while/lstm_cell_116/dropout_3/Mul:z:0&while/lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_3/Mul_1М
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3І
while/lstm_cell_116/mulMulwhile_placeholder_2%while/lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul≠
while/lstm_cell_116/mul_1Mulwhile_placeholder_2'while/lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1≠
while/lstm_cell_116/mul_2Mulwhile_placeholder_2'while/lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2≠
while/lstm_cell_116/mul_3Mulwhile_placeholder_2'while/lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
џж
Ї
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704036

inputsG
4lstm_116_lstm_cell_116_split_readvariableop_resource:	АE
6lstm_116_lstm_cell_116_split_1_readvariableop_resource:	АA
.lstm_116_lstm_cell_116_readvariableop_resource:	 А:
(dense_140_matmul_readvariableop_resource:  7
)dense_140_biasadd_readvariableop_resource: :
(dense_141_matmul_readvariableop_resource: 7
)dense_141_biasadd_readvariableop_resource:
identityИҐ dense_140/BiasAdd/ReadVariableOpҐdense_140/MatMul/ReadVariableOpҐ dense_141/BiasAdd/ReadVariableOpҐdense_141/MatMul/ReadVariableOpҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ%lstm_116/lstm_cell_116/ReadVariableOpҐ'lstm_116/lstm_cell_116/ReadVariableOp_1Ґ'lstm_116/lstm_cell_116/ReadVariableOp_2Ґ'lstm_116/lstm_cell_116/ReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐ+lstm_116/lstm_cell_116/split/ReadVariableOpҐ-lstm_116/lstm_cell_116/split_1/ReadVariableOpҐlstm_116/whileV
lstm_116/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_116/ShapeЖ
lstm_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_116/strided_slice/stackК
lstm_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_116/strided_slice/stack_1К
lstm_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_116/strided_slice/stack_2Ш
lstm_116/strided_sliceStridedSlicelstm_116/Shape:output:0%lstm_116/strided_slice/stack:output:0'lstm_116/strided_slice/stack_1:output:0'lstm_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_116/strided_slicen
lstm_116/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros/mul/yР
lstm_116/zeros/mulMullstm_116/strided_slice:output:0lstm_116/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros/mulq
lstm_116/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_116/zeros/Less/yЛ
lstm_116/zeros/LessLesslstm_116/zeros/mul:z:0lstm_116/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros/Lesst
lstm_116/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros/packed/1І
lstm_116/zeros/packedPacklstm_116/strided_slice:output:0 lstm_116/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_116/zeros/packedq
lstm_116/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/zeros/ConstЩ
lstm_116/zerosFilllstm_116/zeros/packed:output:0lstm_116/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/zerosr
lstm_116/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros_1/mul/yЦ
lstm_116/zeros_1/mulMullstm_116/strided_slice:output:0lstm_116/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros_1/mulu
lstm_116/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_116/zeros_1/Less/yУ
lstm_116/zeros_1/LessLesslstm_116/zeros_1/mul:z:0 lstm_116/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros_1/Lessx
lstm_116/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros_1/packed/1≠
lstm_116/zeros_1/packedPacklstm_116/strided_slice:output:0"lstm_116/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_116/zeros_1/packedu
lstm_116/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/zeros_1/Const°
lstm_116/zeros_1Fill lstm_116/zeros_1/packed:output:0lstm_116/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/zeros_1З
lstm_116/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_116/transpose/permХ
lstm_116/transpose	Transposeinputs lstm_116/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_116/transposej
lstm_116/Shape_1Shapelstm_116/transpose:y:0*
T0*
_output_shapes
:2
lstm_116/Shape_1К
lstm_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_116/strided_slice_1/stackО
 lstm_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_1/stack_1О
 lstm_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_1/stack_2§
lstm_116/strided_slice_1StridedSlicelstm_116/Shape_1:output:0'lstm_116/strided_slice_1/stack:output:0)lstm_116/strided_slice_1/stack_1:output:0)lstm_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_116/strided_slice_1Ч
$lstm_116/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2&
$lstm_116/TensorArrayV2/element_shape÷
lstm_116/TensorArrayV2TensorListReserve-lstm_116/TensorArrayV2/element_shape:output:0!lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_116/TensorArrayV2—
>lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2@
>lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shapeЬ
0lstm_116/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_116/transpose:y:0Glstm_116/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_116/TensorArrayUnstack/TensorListFromTensorК
lstm_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_116/strided_slice_2/stackО
 lstm_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_2/stack_1О
 lstm_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_2/stack_2≤
lstm_116/strided_slice_2StridedSlicelstm_116/transpose:y:0'lstm_116/strided_slice_2/stack:output:0)lstm_116/strided_slice_2/stack_1:output:0)lstm_116/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_116/strided_slice_2Ч
&lstm_116/lstm_cell_116/ones_like/ShapeShapelstm_116/zeros:output:0*
T0*
_output_shapes
:2(
&lstm_116/lstm_cell_116/ones_like/ShapeХ
&lstm_116/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2(
&lstm_116/lstm_cell_116/ones_like/Constа
 lstm_116/lstm_cell_116/ones_likeFill/lstm_116/lstm_cell_116/ones_like/Shape:output:0/lstm_116/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/ones_likeТ
&lstm_116/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_116/lstm_cell_116/split/split_dim–
+lstm_116/lstm_cell_116/split/ReadVariableOpReadVariableOp4lstm_116_lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02-
+lstm_116/lstm_cell_116/split/ReadVariableOpГ
lstm_116/lstm_cell_116/splitSplit/lstm_116/lstm_cell_116/split/split_dim:output:03lstm_116/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_116/lstm_cell_116/splitƒ
lstm_116/lstm_cell_116/MatMulMatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/MatMul»
lstm_116/lstm_cell_116/MatMul_1MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_1»
lstm_116/lstm_cell_116/MatMul_2MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_2»
lstm_116/lstm_cell_116/MatMul_3MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_3Ц
(lstm_116/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_116/lstm_cell_116/split_1/split_dim“
-lstm_116/lstm_cell_116/split_1/ReadVariableOpReadVariableOp6lstm_116_lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-lstm_116/lstm_cell_116/split_1/ReadVariableOpы
lstm_116/lstm_cell_116/split_1Split1lstm_116/lstm_cell_116/split_1/split_dim:output:05lstm_116/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2 
lstm_116/lstm_cell_116/split_1ѕ
lstm_116/lstm_cell_116/BiasAddBiasAdd'lstm_116/lstm_cell_116/MatMul:product:0'lstm_116/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_116/lstm_cell_116/BiasAdd’
 lstm_116/lstm_cell_116/BiasAdd_1BiasAdd)lstm_116/lstm_cell_116/MatMul_1:product:0'lstm_116/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_1’
 lstm_116/lstm_cell_116/BiasAdd_2BiasAdd)lstm_116/lstm_cell_116/MatMul_2:product:0'lstm_116/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_2’
 lstm_116/lstm_cell_116/BiasAdd_3BiasAdd)lstm_116/lstm_cell_116/MatMul_3:product:0'lstm_116/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_3µ
lstm_116/lstm_cell_116/mulMullstm_116/zeros:output:0)lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mulє
lstm_116/lstm_cell_116/mul_1Mullstm_116/zeros:output:0)lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_1є
lstm_116/lstm_cell_116/mul_2Mullstm_116/zeros:output:0)lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_2є
lstm_116/lstm_cell_116/mul_3Mullstm_116/zeros:output:0)lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_3Њ
%lstm_116/lstm_cell_116/ReadVariableOpReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_116/lstm_cell_116/ReadVariableOp©
*lstm_116/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_116/lstm_cell_116/strided_slice/stack≠
,lstm_116/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_116/lstm_cell_116/strided_slice/stack_1≠
,lstm_116/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_116/lstm_cell_116/strided_slice/stack_2Ж
$lstm_116/lstm_cell_116/strided_sliceStridedSlice-lstm_116/lstm_cell_116/ReadVariableOp:value:03lstm_116/lstm_cell_116/strided_slice/stack:output:05lstm_116/lstm_cell_116/strided_slice/stack_1:output:05lstm_116/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_116/lstm_cell_116/strided_sliceЌ
lstm_116/lstm_cell_116/MatMul_4MatMullstm_116/lstm_cell_116/mul:z:0-lstm_116/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_4«
lstm_116/lstm_cell_116/addAddV2'lstm_116/lstm_cell_116/BiasAdd:output:0)lstm_116/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/addЭ
lstm_116/lstm_cell_116/SigmoidSigmoidlstm_116/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_116/lstm_cell_116/Sigmoid¬
'lstm_116/lstm_cell_116/ReadVariableOp_1ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_1≠
,lstm_116/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_116/lstm_cell_116/strided_slice_1/stack±
.lstm_116/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_116/lstm_cell_116/strided_slice_1/stack_1±
.lstm_116/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_1/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_1StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_1:value:05lstm_116/lstm_cell_116/strided_slice_1/stack:output:07lstm_116/lstm_cell_116/strided_slice_1/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_1—
lstm_116/lstm_cell_116/MatMul_5MatMul lstm_116/lstm_cell_116/mul_1:z:0/lstm_116/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_5Ќ
lstm_116/lstm_cell_116/add_1AddV2)lstm_116/lstm_cell_116/BiasAdd_1:output:0)lstm_116/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_1£
 lstm_116/lstm_cell_116/Sigmoid_1Sigmoid lstm_116/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/Sigmoid_1ґ
lstm_116/lstm_cell_116/mul_4Mul$lstm_116/lstm_cell_116/Sigmoid_1:y:0lstm_116/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_4¬
'lstm_116/lstm_cell_116/ReadVariableOp_2ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_2≠
,lstm_116/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_116/lstm_cell_116/strided_slice_2/stack±
.lstm_116/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_116/lstm_cell_116/strided_slice_2/stack_1±
.lstm_116/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_2/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_2StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_2:value:05lstm_116/lstm_cell_116/strided_slice_2/stack:output:07lstm_116/lstm_cell_116/strided_slice_2/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_2—
lstm_116/lstm_cell_116/MatMul_6MatMul lstm_116/lstm_cell_116/mul_2:z:0/lstm_116/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_6Ќ
lstm_116/lstm_cell_116/add_2AddV2)lstm_116/lstm_cell_116/BiasAdd_2:output:0)lstm_116/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_2Ц
lstm_116/lstm_cell_116/ReluRelu lstm_116/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/Reluƒ
lstm_116/lstm_cell_116/mul_5Mul"lstm_116/lstm_cell_116/Sigmoid:y:0)lstm_116/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_5ї
lstm_116/lstm_cell_116/add_3AddV2 lstm_116/lstm_cell_116/mul_4:z:0 lstm_116/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_3¬
'lstm_116/lstm_cell_116/ReadVariableOp_3ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_3≠
,lstm_116/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_116/lstm_cell_116/strided_slice_3/stack±
.lstm_116/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_116/lstm_cell_116/strided_slice_3/stack_1±
.lstm_116/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_3/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_3StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_3:value:05lstm_116/lstm_cell_116/strided_slice_3/stack:output:07lstm_116/lstm_cell_116/strided_slice_3/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_3—
lstm_116/lstm_cell_116/MatMul_7MatMul lstm_116/lstm_cell_116/mul_3:z:0/lstm_116/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_7Ќ
lstm_116/lstm_cell_116/add_4AddV2)lstm_116/lstm_cell_116/BiasAdd_3:output:0)lstm_116/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_4£
 lstm_116/lstm_cell_116/Sigmoid_2Sigmoid lstm_116/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/Sigmoid_2Ъ
lstm_116/lstm_cell_116/Relu_1Relu lstm_116/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/Relu_1»
lstm_116/lstm_cell_116/mul_6Mul$lstm_116/lstm_cell_116/Sigmoid_2:y:0+lstm_116/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_6°
&lstm_116/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2(
&lstm_116/TensorArrayV2_1/element_shape№
lstm_116/TensorArrayV2_1TensorListReserve/lstm_116/TensorArrayV2_1/element_shape:output:0!lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_116/TensorArrayV2_1`
lstm_116/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/timeС
!lstm_116/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2#
!lstm_116/while/maximum_iterations|
lstm_116/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/while/loop_counterН
lstm_116/whileWhile$lstm_116/while/loop_counter:output:0*lstm_116/while/maximum_iterations:output:0lstm_116/time:output:0!lstm_116/TensorArrayV2_1:handle:0lstm_116/zeros:output:0lstm_116/zeros_1:output:0!lstm_116/strided_slice_1:output:0@lstm_116/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_116_lstm_cell_116_split_readvariableop_resource6lstm_116_lstm_cell_116_split_1_readvariableop_resource.lstm_116_lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_116_while_body_3703875*'
condR
lstm_116_while_cond_3703874*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_116/while«
9lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2;
9lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeМ
+lstm_116/TensorArrayV2Stack/TensorListStackTensorListStacklstm_116/while:output:3Blstm_116/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02-
+lstm_116/TensorArrayV2Stack/TensorListStackУ
lstm_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2 
lstm_116/strided_slice_3/stackО
 lstm_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_116/strided_slice_3/stack_1О
 lstm_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_3/stack_2–
lstm_116/strided_slice_3StridedSlice4lstm_116/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_116/strided_slice_3/stack:output:0)lstm_116/strided_slice_3/stack_1:output:0)lstm_116/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_116/strided_slice_3Л
lstm_116/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_116/transpose_1/perm…
lstm_116/transpose_1	Transpose4lstm_116/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_116/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_116/transpose_1x
lstm_116/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/runtimeЂ
dense_140/MatMul/ReadVariableOpReadVariableOp(dense_140_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense_140/MatMul/ReadVariableOpђ
dense_140/MatMulMatMul!lstm_116/strided_slice_3:output:0'dense_140/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/MatMul™
 dense_140/BiasAdd/ReadVariableOpReadVariableOp)dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_140/BiasAdd/ReadVariableOp©
dense_140/BiasAddBiasAdddense_140/MatMul:product:0(dense_140/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/BiasAddv
dense_140/ReluReludense_140/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/ReluЂ
dense_141/MatMul/ReadVariableOpReadVariableOp(dense_141_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense_141/MatMul/ReadVariableOpІ
dense_141/MatMulMatMuldense_140/Relu:activations:0'dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_141/MatMul™
 dense_141/BiasAdd/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_141/BiasAdd/ReadVariableOp©
dense_141/BiasAddBiasAdddense_141/MatMul:product:0(dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_141/BiasAddn
reshape_70/ShapeShapedense_141/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_70/ShapeК
reshape_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_70/strided_slice/stackО
 reshape_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_70/strided_slice/stack_1О
 reshape_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_70/strided_slice/stack_2§
reshape_70/strided_sliceStridedSlicereshape_70/Shape:output:0'reshape_70/strided_slice/stack:output:0)reshape_70/strided_slice/stack_1:output:0)reshape_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_70/strided_slicez
reshape_70/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_70/Reshape/shape/1z
reshape_70/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_70/Reshape/shape/2„
reshape_70/Reshape/shapePack!reshape_70/strided_slice:output:0#reshape_70/Reshape/shape/1:output:0#reshape_70/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_70/Reshape/shape®
reshape_70/ReshapeReshapedense_141/BiasAdd:output:0!reshape_70/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_70/Reshapeш
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_116_lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul 
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulz
IdentityIdentityreshape_70/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityв
NoOpNoOp!^dense_140/BiasAdd/ReadVariableOp ^dense_140/MatMul/ReadVariableOp!^dense_141/BiasAdd/ReadVariableOp ^dense_141/MatMul/ReadVariableOp1^dense_141/bias/Regularizer/Square/ReadVariableOp&^lstm_116/lstm_cell_116/ReadVariableOp(^lstm_116/lstm_cell_116/ReadVariableOp_1(^lstm_116/lstm_cell_116/ReadVariableOp_2(^lstm_116/lstm_cell_116/ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp,^lstm_116/lstm_cell_116/split/ReadVariableOp.^lstm_116/lstm_cell_116/split_1/ReadVariableOp^lstm_116/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_140/BiasAdd/ReadVariableOp dense_140/BiasAdd/ReadVariableOp2B
dense_140/MatMul/ReadVariableOpdense_140/MatMul/ReadVariableOp2D
 dense_141/BiasAdd/ReadVariableOp dense_141/BiasAdd/ReadVariableOp2B
dense_141/MatMul/ReadVariableOpdense_141/MatMul/ReadVariableOp2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2N
%lstm_116/lstm_cell_116/ReadVariableOp%lstm_116/lstm_cell_116/ReadVariableOp2R
'lstm_116/lstm_cell_116/ReadVariableOp_1'lstm_116/lstm_cell_116/ReadVariableOp_12R
'lstm_116/lstm_cell_116/ReadVariableOp_2'lstm_116/lstm_cell_116/ReadVariableOp_22R
'lstm_116/lstm_cell_116/ReadVariableOp_3'lstm_116/lstm_cell_116/ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2Z
+lstm_116/lstm_cell_116/split/ReadVariableOp+lstm_116/lstm_cell_116/split/ReadVariableOp2^
-lstm_116/lstm_cell_116/split_1/ReadVariableOp-lstm_116/lstm_cell_116/split_1/ReadVariableOp2 
lstm_116/whilelstm_116/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В&
с
while_body_3702187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_116_3702211_0:	А,
while_lstm_cell_116_3702213_0:	А0
while_lstm_cell_116_3702215_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_116_3702211:	А*
while_lstm_cell_116_3702213:	А.
while_lstm_cell_116_3702215:	 АИҐ+while/lstm_cell_116/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemл
+while/lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_116_3702211_0while_lstm_cell_116_3702213_0while_lstm_cell_116_3702215_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37021732-
+while/lstm_cell_116/StatefulPartitionedCallш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_116/StatefulPartitionedCall:output:0*
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
while/Identity_3•
while/Identity_4Identity4while/lstm_cell_116/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4•
while/Identity_5Identity4while/lstm_cell_116/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5И

while/NoOpNoOp,^while/lstm_cell_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_116_3702211while_lstm_cell_116_3702211_0"<
while_lstm_cell_116_3702213while_lstm_cell_116_3702213_0"<
while_lstm_cell_116_3702215while_lstm_cell_116_3702215_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2Z
+while/lstm_cell_116/StatefulPartitionedCall+while/lstm_cell_116/StatefulPartitionedCall: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
„R
–
E__inference_lstm_116_layer_call_and_return_conditional_losses_3702559

inputs(
lstm_cell_116_3702471:	А$
lstm_cell_116_3702473:	А(
lstm_cell_116_3702475:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐ%lstm_cell_116/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2І
%lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_116_3702471lstm_cell_116_3702473lstm_cell_116_3702475*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37024062'
%lstm_cell_116/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter»
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_116_3702471lstm_cell_116_3702473lstm_cell_116_3702475*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3702484*
condR
while_cond_3702483*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeў
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_116_3702471*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityј
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp&^lstm_cell_116/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2N
%lstm_cell_116/StatefulPartitionedCall%lstm_cell_116/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Є	
†
%__inference_signature_wrapper_3703727
input_48
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_37020492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
√љ
ў
)sequential_47_lstm_116_while_body_3701900J
Fsequential_47_lstm_116_while_sequential_47_lstm_116_while_loop_counterP
Lsequential_47_lstm_116_while_sequential_47_lstm_116_while_maximum_iterations,
(sequential_47_lstm_116_while_placeholder.
*sequential_47_lstm_116_while_placeholder_1.
*sequential_47_lstm_116_while_placeholder_2.
*sequential_47_lstm_116_while_placeholder_3I
Esequential_47_lstm_116_while_sequential_47_lstm_116_strided_slice_1_0Ж
Бsequential_47_lstm_116_while_tensorarrayv2read_tensorlistgetitem_sequential_47_lstm_116_tensorarrayunstack_tensorlistfromtensor_0]
Jsequential_47_lstm_116_while_lstm_cell_116_split_readvariableop_resource_0:	А[
Lsequential_47_lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0:	АW
Dsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0:	 А)
%sequential_47_lstm_116_while_identity+
'sequential_47_lstm_116_while_identity_1+
'sequential_47_lstm_116_while_identity_2+
'sequential_47_lstm_116_while_identity_3+
'sequential_47_lstm_116_while_identity_4+
'sequential_47_lstm_116_while_identity_5G
Csequential_47_lstm_116_while_sequential_47_lstm_116_strided_slice_1Г
sequential_47_lstm_116_while_tensorarrayv2read_tensorlistgetitem_sequential_47_lstm_116_tensorarrayunstack_tensorlistfromtensor[
Hsequential_47_lstm_116_while_lstm_cell_116_split_readvariableop_resource:	АY
Jsequential_47_lstm_116_while_lstm_cell_116_split_1_readvariableop_resource:	АU
Bsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource:	 АИҐ9sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOpҐ;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1Ґ;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2Ґ;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3Ґ?sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOpҐAsequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOpс
Nsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2P
Nsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeё
@sequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemБsequential_47_lstm_116_while_tensorarrayv2read_tensorlistgetitem_sequential_47_lstm_116_tensorarrayunstack_tensorlistfromtensor_0(sequential_47_lstm_116_while_placeholderWsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02B
@sequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem“
:sequential_47/lstm_116/while/lstm_cell_116/ones_like/ShapeShape*sequential_47_lstm_116_while_placeholder_2*
T0*
_output_shapes
:2<
:sequential_47/lstm_116/while/lstm_cell_116/ones_like/Shapeљ
:sequential_47/lstm_116/while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2<
:sequential_47/lstm_116/while/lstm_cell_116/ones_like/Const∞
4sequential_47/lstm_116/while/lstm_cell_116/ones_likeFillCsequential_47/lstm_116/while/lstm_cell_116/ones_like/Shape:output:0Csequential_47/lstm_116/while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/ones_likeЇ
:sequential_47/lstm_116/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_47/lstm_116/while/lstm_cell_116/split/split_dimО
?sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOpReadVariableOpJsequential_47_lstm_116_while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02A
?sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOp”
0sequential_47/lstm_116/while/lstm_cell_116/splitSplitCsequential_47/lstm_116/while/lstm_cell_116/split/split_dim:output:0Gsequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split22
0sequential_47/lstm_116/while/lstm_cell_116/split¶
1sequential_47/lstm_116/while/lstm_cell_116/MatMulMatMulGsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_47/lstm_116/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_47/lstm_116/while/lstm_cell_116/MatMul™
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_1MatMulGsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_47/lstm_116/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_1™
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_2MatMulGsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_47/lstm_116/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_2™
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_3MatMulGsequential_47/lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_47/lstm_116/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_3Њ
<sequential_47/lstm_116/while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_47/lstm_116/while/lstm_cell_116/split_1/split_dimР
Asequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOpReadVariableOpLsequential_47_lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02C
Asequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOpЋ
2sequential_47/lstm_116/while/lstm_cell_116/split_1SplitEsequential_47/lstm_116/while/lstm_cell_116/split_1/split_dim:output:0Isequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split24
2sequential_47/lstm_116/while/lstm_cell_116/split_1Я
2sequential_47/lstm_116/while/lstm_cell_116/BiasAddBiasAdd;sequential_47/lstm_116/while/lstm_cell_116/MatMul:product:0;sequential_47/lstm_116/while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 24
2sequential_47/lstm_116/while/lstm_cell_116/BiasAdd•
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_1BiasAdd=sequential_47/lstm_116/while/lstm_cell_116/MatMul_1:product:0;sequential_47/lstm_116/while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_1•
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_2BiasAdd=sequential_47/lstm_116/while/lstm_cell_116/MatMul_2:product:0;sequential_47/lstm_116/while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_2•
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_3BiasAdd=sequential_47/lstm_116/while/lstm_cell_116/MatMul_3:product:0;sequential_47/lstm_116/while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_3Д
.sequential_47/lstm_116/while/lstm_cell_116/mulMul*sequential_47_lstm_116_while_placeholder_2=sequential_47/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/while/lstm_cell_116/mulИ
0sequential_47/lstm_116/while/lstm_cell_116/mul_1Mul*sequential_47_lstm_116_while_placeholder_2=sequential_47/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_1И
0sequential_47/lstm_116/while/lstm_cell_116/mul_2Mul*sequential_47_lstm_116_while_placeholder_2=sequential_47/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_2И
0sequential_47/lstm_116/while/lstm_cell_116/mul_3Mul*sequential_47_lstm_116_while_placeholder_2=sequential_47/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_3ь
9sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOpReadVariableOpDsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02;
9sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp—
>sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack’
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_1’
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_2ю
8sequential_47/lstm_116/while/lstm_cell_116/strided_sliceStridedSliceAsequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp:value:0Gsequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack:output:0Isequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_1:output:0Isequential_47/lstm_116/while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_47/lstm_116/while/lstm_cell_116/strided_sliceЭ
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_4MatMul2sequential_47/lstm_116/while/lstm_cell_116/mul:z:0Asequential_47/lstm_116/while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_4Ч
.sequential_47/lstm_116/while/lstm_cell_116/addAddV2;sequential_47/lstm_116/while/lstm_cell_116/BiasAdd:output:0=sequential_47/lstm_116/while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/while/lstm_cell_116/addў
2sequential_47/lstm_116/while/lstm_cell_116/SigmoidSigmoid2sequential_47/lstm_116/while/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 24
2sequential_47/lstm_116/while/lstm_cell_116/SigmoidА
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1ReadVariableOpDsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02=
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1’
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stackў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_1ў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_2К
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_1StridedSliceCsequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1:value:0Isequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_1:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_1°
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_5MatMul4sequential_47/lstm_116/while/lstm_cell_116/mul_1:z:0Csequential_47/lstm_116/while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_5Э
0sequential_47/lstm_116/while/lstm_cell_116/add_1AddV2=sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_1:output:0=sequential_47/lstm_116/while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/add_1я
4sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_1Sigmoid4sequential_47/lstm_116/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_1Г
0sequential_47/lstm_116/while/lstm_cell_116/mul_4Mul8sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_1:y:0*sequential_47_lstm_116_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_4А
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2ReadVariableOpDsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02=
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2’
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2B
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stackў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_1ў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_2К
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_2StridedSliceCsequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2:value:0Isequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_1:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_2°
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_6MatMul4sequential_47/lstm_116/while/lstm_cell_116/mul_2:z:0Csequential_47/lstm_116/while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_6Э
0sequential_47/lstm_116/while/lstm_cell_116/add_2AddV2=sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_2:output:0=sequential_47/lstm_116/while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/add_2“
/sequential_47/lstm_116/while/lstm_cell_116/ReluRelu4sequential_47/lstm_116/while/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_47/lstm_116/while/lstm_cell_116/ReluФ
0sequential_47/lstm_116/while/lstm_cell_116/mul_5Mul6sequential_47/lstm_116/while/lstm_cell_116/Sigmoid:y:0=sequential_47/lstm_116/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_5Л
0sequential_47/lstm_116/while/lstm_cell_116/add_3AddV24sequential_47/lstm_116/while/lstm_cell_116/mul_4:z:04sequential_47/lstm_116/while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/add_3А
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3ReadVariableOpDsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02=
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3’
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2B
@sequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stackў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_1ў
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_2К
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_3StridedSliceCsequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3:value:0Isequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_1:output:0Ksequential_47/lstm_116/while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_47/lstm_116/while/lstm_cell_116/strided_slice_3°
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_7MatMul4sequential_47/lstm_116/while/lstm_cell_116/mul_3:z:0Csequential_47/lstm_116/while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3sequential_47/lstm_116/while/lstm_cell_116/MatMul_7Э
0sequential_47/lstm_116/while/lstm_cell_116/add_4AddV2=sequential_47/lstm_116/while/lstm_cell_116/BiasAdd_3:output:0=sequential_47/lstm_116/while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/add_4я
4sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_2Sigmoid4sequential_47/lstm_116/while/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 26
4sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_2÷
1sequential_47/lstm_116/while/lstm_cell_116/Relu_1Relu4sequential_47/lstm_116/while/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_47/lstm_116/while/lstm_cell_116/Relu_1Ш
0sequential_47/lstm_116/while/lstm_cell_116/mul_6Mul8sequential_47/lstm_116/while/lstm_cell_116/Sigmoid_2:y:0?sequential_47/lstm_116/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_47/lstm_116/while/lstm_cell_116/mul_6‘
Asequential_47/lstm_116/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_47_lstm_116_while_placeholder_1(sequential_47_lstm_116_while_placeholder4sequential_47/lstm_116/while/lstm_cell_116/mul_6:z:0*
_output_shapes
: *
element_dtype02C
Asequential_47/lstm_116/while/TensorArrayV2Write/TensorListSetItemК
"sequential_47/lstm_116/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_47/lstm_116/while/add/y≈
 sequential_47/lstm_116/while/addAddV2(sequential_47_lstm_116_while_placeholder+sequential_47/lstm_116/while/add/y:output:0*
T0*
_output_shapes
: 2"
 sequential_47/lstm_116/while/addО
$sequential_47/lstm_116/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_47/lstm_116/while/add_1/yй
"sequential_47/lstm_116/while/add_1AddV2Fsequential_47_lstm_116_while_sequential_47_lstm_116_while_loop_counter-sequential_47/lstm_116/while/add_1/y:output:0*
T0*
_output_shapes
: 2$
"sequential_47/lstm_116/while/add_1«
%sequential_47/lstm_116/while/IdentityIdentity&sequential_47/lstm_116/while/add_1:z:0"^sequential_47/lstm_116/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_47/lstm_116/while/Identityс
'sequential_47/lstm_116/while/Identity_1IdentityLsequential_47_lstm_116_while_sequential_47_lstm_116_while_maximum_iterations"^sequential_47/lstm_116/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_47/lstm_116/while/Identity_1…
'sequential_47/lstm_116/while/Identity_2Identity$sequential_47/lstm_116/while/add:z:0"^sequential_47/lstm_116/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_47/lstm_116/while/Identity_2ц
'sequential_47/lstm_116/while/Identity_3IdentityQsequential_47/lstm_116/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_47/lstm_116/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_47/lstm_116/while/Identity_3к
'sequential_47/lstm_116/while/Identity_4Identity4sequential_47/lstm_116/while/lstm_cell_116/mul_6:z:0"^sequential_47/lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_47/lstm_116/while/Identity_4к
'sequential_47/lstm_116/while/Identity_5Identity4sequential_47/lstm_116/while/lstm_cell_116/add_3:z:0"^sequential_47/lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_47/lstm_116/while/Identity_5Д
!sequential_47/lstm_116/while/NoOpNoOp:^sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp<^sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1<^sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2<^sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3@^sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOpB^sequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2#
!sequential_47/lstm_116/while/NoOp"W
%sequential_47_lstm_116_while_identity.sequential_47/lstm_116/while/Identity:output:0"[
'sequential_47_lstm_116_while_identity_10sequential_47/lstm_116/while/Identity_1:output:0"[
'sequential_47_lstm_116_while_identity_20sequential_47/lstm_116/while/Identity_2:output:0"[
'sequential_47_lstm_116_while_identity_30sequential_47/lstm_116/while/Identity_3:output:0"[
'sequential_47_lstm_116_while_identity_40sequential_47/lstm_116/while/Identity_4:output:0"[
'sequential_47_lstm_116_while_identity_50sequential_47/lstm_116/while/Identity_5:output:0"К
Bsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resourceDsequential_47_lstm_116_while_lstm_cell_116_readvariableop_resource_0"Ъ
Jsequential_47_lstm_116_while_lstm_cell_116_split_1_readvariableop_resourceLsequential_47_lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0"Ц
Hsequential_47_lstm_116_while_lstm_cell_116_split_readvariableop_resourceJsequential_47_lstm_116_while_lstm_cell_116_split_readvariableop_resource_0"М
Csequential_47_lstm_116_while_sequential_47_lstm_116_strided_slice_1Esequential_47_lstm_116_while_sequential_47_lstm_116_strided_slice_1_0"Е
sequential_47_lstm_116_while_tensorarrayv2read_tensorlistgetitem_sequential_47_lstm_116_tensorarrayunstack_tensorlistfromtensorБsequential_47_lstm_116_while_tensorarrayv2read_tensorlistgetitem_sequential_47_lstm_116_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2v
9sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp9sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp2z
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_1;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_12z
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_2;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_22z
;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_3;sequential_47/lstm_116/while/lstm_cell_116/ReadVariableOp_32В
?sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOp?sequential_47/lstm_116/while/lstm_cell_116/split/ReadVariableOp2Ж
Asequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOpAsequential_47/lstm_116/while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
к	
™
/__inference_sequential_47_layer_call_fn_3703174
input_48
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_47_layer_call_and_return_conditional_losses_37031572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
‘v
н
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3702406

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
:€€€€€€€€€ 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape—
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2йе©2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_1/ConstЕ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape„
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2жОђ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_1/GreaterEqual/y∆
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_2/ConstЕ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape„
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2÷†Њ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_2/GreaterEqual/y∆
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_3/ConstЕ
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape„
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2—™и2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_3/GreaterEqual/y∆
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
:€€€€€€€€€ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_3^
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
muld
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1d
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice/stack_2ь
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
:€€€€€€€€€ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
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
:€€€€€€€€€ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6б
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_2К
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
Џ
»
while_cond_3702186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3702186___redundant_placeholder05
1while_while_cond_3702186___redundant_placeholder15
1while_while_cond_3702186___redundant_placeholder25
1while_while_cond_3702186___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
к	
™
/__inference_sequential_47_layer_call_fn_3703620
input_48
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_47_layer_call_and_return_conditional_losses_37035842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
»“
і
E__inference_lstm_116_layer_call_and_return_conditional_losses_3703520

inputs>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
:€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_like
lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout/ConstЈ
lstm_cell_116/dropout/MulMul lstm_cell_116/ones_like:output:0$lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/MulК
lstm_cell_116/dropout/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout/Shapeы
2lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2к«≥24
2lstm_cell_116/dropout/random_uniform/RandomUniformС
$lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2&
$lstm_cell_116/dropout/GreaterEqual/yц
"lstm_cell_116/dropout/GreaterEqualGreaterEqual;lstm_cell_116/dropout/random_uniform/RandomUniform:output:0-lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_116/dropout/GreaterEqual©
lstm_cell_116/dropout/CastCast&lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Cast≤
lstm_cell_116/dropout/Mul_1Mullstm_cell_116/dropout/Mul:z:0lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Mul_1Г
lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_1/Constљ
lstm_cell_116/dropout_1/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/MulО
lstm_cell_116/dropout_1/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_1/ShapeА
4lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЬА26
4lstm_cell_116/dropout_1/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_1/GreaterEqual/yю
$lstm_cell_116/dropout_1/GreaterEqualGreaterEqual=lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_1/GreaterEqualѓ
lstm_cell_116/dropout_1/CastCast(lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/CastЇ
lstm_cell_116/dropout_1/Mul_1Mullstm_cell_116/dropout_1/Mul:z:0 lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/Mul_1Г
lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_2/Constљ
lstm_cell_116/dropout_2/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/MulО
lstm_cell_116/dropout_2/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_2/ShapeБ
4lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ќЂ 26
4lstm_cell_116/dropout_2/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_2/GreaterEqual/yю
$lstm_cell_116/dropout_2/GreaterEqualGreaterEqual=lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_2/GreaterEqualѓ
lstm_cell_116/dropout_2/CastCast(lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/CastЇ
lstm_cell_116/dropout_2/Mul_1Mullstm_cell_116/dropout_2/Mul:z:0 lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/Mul_1Г
lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_3/Constљ
lstm_cell_116/dropout_3/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/MulО
lstm_cell_116/dropout_3/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_3/ShapeБ
4lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ХђЙ26
4lstm_cell_116/dropout_3/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_3/GreaterEqual/yю
$lstm_cell_116/dropout_3/GreaterEqualGreaterEqual=lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_3/GreaterEqualѓ
lstm_cell_116/dropout_3/CastCast(lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/CastЇ
lstm_cell_116/dropout_3/Mul_1Mullstm_cell_116/dropout_3/Mul:z:0 lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/Mul_1А
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3Р
lstm_cell_116/mulMulzeros:output:0lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulЦ
lstm_cell_116/mul_1Mulzeros:output:0!lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Ц
lstm_cell_116/mul_2Mulzeros:output:0!lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Ц
lstm_cell_116/mul_3Mulzeros:output:0!lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3703355*
condR
while_cond_3703354*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm•
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
О
ѕ
__inference_loss_fn_1_3705846[
Hlstm_116_lstm_cell_116_kernel_regularizer_square_readvariableop_resource:	А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpМ
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHlstm_116_lstm_cell_116_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul{
IdentityIdentity1lstm_116/lstm_cell_116/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityР
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp
О
Ф
)sequential_47_lstm_116_while_cond_3701899J
Fsequential_47_lstm_116_while_sequential_47_lstm_116_while_loop_counterP
Lsequential_47_lstm_116_while_sequential_47_lstm_116_while_maximum_iterations,
(sequential_47_lstm_116_while_placeholder.
*sequential_47_lstm_116_while_placeholder_1.
*sequential_47_lstm_116_while_placeholder_2.
*sequential_47_lstm_116_while_placeholder_3L
Hsequential_47_lstm_116_while_less_sequential_47_lstm_116_strided_slice_1c
_sequential_47_lstm_116_while_sequential_47_lstm_116_while_cond_3701899___redundant_placeholder0c
_sequential_47_lstm_116_while_sequential_47_lstm_116_while_cond_3701899___redundant_placeholder1c
_sequential_47_lstm_116_while_sequential_47_lstm_116_while_cond_3701899___redundant_placeholder2c
_sequential_47_lstm_116_while_sequential_47_lstm_116_while_cond_3701899___redundant_placeholder3)
%sequential_47_lstm_116_while_identity
г
!sequential_47/lstm_116/while/LessLess(sequential_47_lstm_116_while_placeholderHsequential_47_lstm_116_while_less_sequential_47_lstm_116_strided_slice_1*
T0*
_output_shapes
: 2#
!sequential_47/lstm_116/while/LessҐ
%sequential_47/lstm_116/while/IdentityIdentity%sequential_47/lstm_116/while/Less:z:0*
T0
*
_output_shapes
: 2'
%sequential_47/lstm_116/while/Identity"W
%sequential_47_lstm_116_while_identity.sequential_47/lstm_116/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ыR
н
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3702173

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
:€€€€€€€€€ 2
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
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
:€€€€€€€€€ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_3_
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulc
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1c
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2c
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice/stack_2ь
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
:€€€€€€€€€ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
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
:€€€€€€€€€ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6б
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_2К
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
д”
м
lstm_116_while_body_3704178.
*lstm_116_while_lstm_116_while_loop_counter4
0lstm_116_while_lstm_116_while_maximum_iterations
lstm_116_while_placeholder 
lstm_116_while_placeholder_1 
lstm_116_while_placeholder_2 
lstm_116_while_placeholder_3-
)lstm_116_while_lstm_116_strided_slice_1_0i
elstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0:	АM
>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0:	АI
6lstm_116_while_lstm_cell_116_readvariableop_resource_0:	 А
lstm_116_while_identity
lstm_116_while_identity_1
lstm_116_while_identity_2
lstm_116_while_identity_3
lstm_116_while_identity_4
lstm_116_while_identity_5+
'lstm_116_while_lstm_116_strided_slice_1g
clstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensorM
:lstm_116_while_lstm_cell_116_split_readvariableop_resource:	АK
<lstm_116_while_lstm_cell_116_split_1_readvariableop_resource:	АG
4lstm_116_while_lstm_cell_116_readvariableop_resource:	 АИҐ+lstm_116/while/lstm_cell_116/ReadVariableOpҐ-lstm_116/while/lstm_cell_116/ReadVariableOp_1Ґ-lstm_116/while/lstm_cell_116/ReadVariableOp_2Ґ-lstm_116/while/lstm_cell_116/ReadVariableOp_3Ґ1lstm_116/while/lstm_cell_116/split/ReadVariableOpҐ3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp’
@lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2B
@lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
2lstm_116/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0lstm_116_while_placeholderIlstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype024
2lstm_116/while/TensorArrayV2Read/TensorListGetItem®
,lstm_116/while/lstm_cell_116/ones_like/ShapeShapelstm_116_while_placeholder_2*
T0*
_output_shapes
:2.
,lstm_116/while/lstm_cell_116/ones_like/Shape°
,lstm_116/while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,lstm_116/while/lstm_cell_116/ones_like/Constш
&lstm_116/while/lstm_cell_116/ones_likeFill5lstm_116/while/lstm_cell_116/ones_like/Shape:output:05lstm_116/while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/ones_likeЭ
*lstm_116/while/lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2,
*lstm_116/while/lstm_cell_116/dropout/Constу
(lstm_116/while/lstm_cell_116/dropout/MulMul/lstm_116/while/lstm_cell_116/ones_like:output:03lstm_116/while/lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_116/while/lstm_cell_116/dropout/MulЈ
*lstm_116/while/lstm_cell_116/dropout/ShapeShape/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_116/while/lstm_cell_116/dropout/ShapeІ
Alstm_116/while/lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform3lstm_116/while/lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Њі.2C
Alstm_116/while/lstm_cell_116/dropout/random_uniform/RandomUniformѓ
3lstm_116/while/lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_116/while/lstm_cell_116/dropout/GreaterEqual/y≤
1lstm_116/while/lstm_cell_116/dropout/GreaterEqualGreaterEqualJlstm_116/while/lstm_cell_116/dropout/random_uniform/RandomUniform:output:0<lstm_116/while/lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1lstm_116/while/lstm_cell_116/dropout/GreaterEqual÷
)lstm_116/while/lstm_cell_116/dropout/CastCast5lstm_116/while/lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_116/while/lstm_cell_116/dropout/Castо
*lstm_116/while/lstm_cell_116/dropout/Mul_1Mul,lstm_116/while/lstm_cell_116/dropout/Mul:z:0-lstm_116/while/lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_116/while/lstm_cell_116/dropout/Mul_1°
,lstm_116/while/lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2.
,lstm_116/while/lstm_cell_116/dropout_1/Constщ
*lstm_116/while/lstm_cell_116/dropout_1/MulMul/lstm_116/while/lstm_cell_116/ones_like:output:05lstm_116/while/lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_116/while/lstm_cell_116/dropout_1/Mulї
,lstm_116/while/lstm_cell_116/dropout_1/ShapeShape/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_116/while/lstm_cell_116/dropout_1/ShapeЃ
Clstm_116/while/lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform5lstm_116/while/lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЄЪа2E
Clstm_116/while/lstm_cell_116/dropout_1/random_uniform/RandomUniform≥
5lstm_116/while/lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>27
5lstm_116/while/lstm_cell_116/dropout_1/GreaterEqual/yЇ
3lstm_116/while/lstm_cell_116/dropout_1/GreaterEqualGreaterEqualLlstm_116/while/lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:0>lstm_116/while/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3lstm_116/while/lstm_cell_116/dropout_1/GreaterEqual№
+lstm_116/while/lstm_cell_116/dropout_1/CastCast7lstm_116/while/lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_116/while/lstm_cell_116/dropout_1/Castц
,lstm_116/while/lstm_cell_116/dropout_1/Mul_1Mul.lstm_116/while/lstm_cell_116/dropout_1/Mul:z:0/lstm_116/while/lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,lstm_116/while/lstm_cell_116/dropout_1/Mul_1°
,lstm_116/while/lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2.
,lstm_116/while/lstm_cell_116/dropout_2/Constщ
*lstm_116/while/lstm_cell_116/dropout_2/MulMul/lstm_116/while/lstm_cell_116/ones_like:output:05lstm_116/while/lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_116/while/lstm_cell_116/dropout_2/Mulї
,lstm_116/while/lstm_cell_116/dropout_2/ShapeShape/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_116/while/lstm_cell_116/dropout_2/ShapeЃ
Clstm_116/while/lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform5lstm_116/while/lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЉД£2E
Clstm_116/while/lstm_cell_116/dropout_2/random_uniform/RandomUniform≥
5lstm_116/while/lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>27
5lstm_116/while/lstm_cell_116/dropout_2/GreaterEqual/yЇ
3lstm_116/while/lstm_cell_116/dropout_2/GreaterEqualGreaterEqualLlstm_116/while/lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:0>lstm_116/while/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3lstm_116/while/lstm_cell_116/dropout_2/GreaterEqual№
+lstm_116/while/lstm_cell_116/dropout_2/CastCast7lstm_116/while/lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_116/while/lstm_cell_116/dropout_2/Castц
,lstm_116/while/lstm_cell_116/dropout_2/Mul_1Mul.lstm_116/while/lstm_cell_116/dropout_2/Mul:z:0/lstm_116/while/lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,lstm_116/while/lstm_cell_116/dropout_2/Mul_1°
,lstm_116/while/lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2.
,lstm_116/while/lstm_cell_116/dropout_3/Constщ
*lstm_116/while/lstm_cell_116/dropout_3/MulMul/lstm_116/while/lstm_cell_116/ones_like:output:05lstm_116/while/lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_116/while/lstm_cell_116/dropout_3/Mulї
,lstm_116/while/lstm_cell_116/dropout_3/ShapeShape/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_116/while/lstm_cell_116/dropout_3/Shape≠
Clstm_116/while/lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform5lstm_116/while/lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ё™(2E
Clstm_116/while/lstm_cell_116/dropout_3/random_uniform/RandomUniform≥
5lstm_116/while/lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>27
5lstm_116/while/lstm_cell_116/dropout_3/GreaterEqual/yЇ
3lstm_116/while/lstm_cell_116/dropout_3/GreaterEqualGreaterEqualLlstm_116/while/lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:0>lstm_116/while/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 25
3lstm_116/while/lstm_cell_116/dropout_3/GreaterEqual№
+lstm_116/while/lstm_cell_116/dropout_3/CastCast7lstm_116/while/lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_116/while/lstm_cell_116/dropout_3/Castц
,lstm_116/while/lstm_cell_116/dropout_3/Mul_1Mul.lstm_116/while/lstm_cell_116/dropout_3/Mul:z:0/lstm_116/while/lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,lstm_116/while/lstm_cell_116/dropout_3/Mul_1Ю
,lstm_116/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_116/while/lstm_cell_116/split/split_dimд
1lstm_116/while/lstm_cell_116/split/ReadVariableOpReadVariableOp<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype023
1lstm_116/while/lstm_cell_116/split/ReadVariableOpЫ
"lstm_116/while/lstm_cell_116/splitSplit5lstm_116/while/lstm_cell_116/split/split_dim:output:09lstm_116/while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2$
"lstm_116/while/lstm_cell_116/splitо
#lstm_116/while/lstm_cell_116/MatMulMatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_116/while/lstm_cell_116/MatMulт
%lstm_116/while/lstm_cell_116/MatMul_1MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_1т
%lstm_116/while/lstm_cell_116/MatMul_2MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_2т
%lstm_116/while/lstm_cell_116/MatMul_3MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_3Ґ
.lstm_116/while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.lstm_116/while/lstm_cell_116/split_1/split_dimж
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype025
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOpУ
$lstm_116/while/lstm_cell_116/split_1Split7lstm_116/while/lstm_cell_116/split_1/split_dim:output:0;lstm_116/while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2&
$lstm_116/while/lstm_cell_116/split_1з
$lstm_116/while/lstm_cell_116/BiasAddBiasAdd-lstm_116/while/lstm_cell_116/MatMul:product:0-lstm_116/while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/while/lstm_cell_116/BiasAddн
&lstm_116/while/lstm_cell_116/BiasAdd_1BiasAdd/lstm_116/while/lstm_cell_116/MatMul_1:product:0-lstm_116/while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_1н
&lstm_116/while/lstm_cell_116/BiasAdd_2BiasAdd/lstm_116/while/lstm_cell_116/MatMul_2:product:0-lstm_116/while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_2н
&lstm_116/while/lstm_cell_116/BiasAdd_3BiasAdd/lstm_116/while/lstm_cell_116/MatMul_3:product:0-lstm_116/while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_3Ћ
 lstm_116/while/lstm_cell_116/mulMullstm_116_while_placeholder_2.lstm_116/while/lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/while/lstm_cell_116/mul—
"lstm_116/while/lstm_cell_116/mul_1Mullstm_116_while_placeholder_20lstm_116/while/lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_1—
"lstm_116/while/lstm_cell_116/mul_2Mullstm_116_while_placeholder_20lstm_116/while/lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_2—
"lstm_116/while/lstm_cell_116/mul_3Mullstm_116_while_placeholder_20lstm_116/while/lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_3“
+lstm_116/while/lstm_cell_116/ReadVariableOpReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_116/while/lstm_cell_116/ReadVariableOpµ
0lstm_116/while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_116/while/lstm_cell_116/strided_slice/stackє
2lstm_116/while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_116/while/lstm_cell_116/strided_slice/stack_1є
2lstm_116/while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_116/while/lstm_cell_116/strided_slice/stack_2™
*lstm_116/while/lstm_cell_116/strided_sliceStridedSlice3lstm_116/while/lstm_cell_116/ReadVariableOp:value:09lstm_116/while/lstm_cell_116/strided_slice/stack:output:0;lstm_116/while/lstm_cell_116/strided_slice/stack_1:output:0;lstm_116/while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_116/while/lstm_cell_116/strided_sliceе
%lstm_116/while/lstm_cell_116/MatMul_4MatMul$lstm_116/while/lstm_cell_116/mul:z:03lstm_116/while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_4я
 lstm_116/while/lstm_cell_116/addAddV2-lstm_116/while/lstm_cell_116/BiasAdd:output:0/lstm_116/while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/while/lstm_cell_116/addѓ
$lstm_116/while/lstm_cell_116/SigmoidSigmoid$lstm_116/while/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/while/lstm_cell_116/Sigmoid÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_1ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_1є
2lstm_116/while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_116/while/lstm_cell_116/strided_slice_1/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   26
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_1StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_1:value:0;lstm_116/while/lstm_cell_116/strided_slice_1/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_1/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_1й
%lstm_116/while/lstm_cell_116/MatMul_5MatMul&lstm_116/while/lstm_cell_116/mul_1:z:05lstm_116/while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_5е
"lstm_116/while/lstm_cell_116/add_1AddV2/lstm_116/while/lstm_cell_116/BiasAdd_1:output:0/lstm_116/while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_1µ
&lstm_116/while/lstm_cell_116/Sigmoid_1Sigmoid&lstm_116/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/Sigmoid_1Ћ
"lstm_116/while/lstm_cell_116/mul_4Mul*lstm_116/while/lstm_cell_116/Sigmoid_1:y:0lstm_116_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_4÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_2ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_2є
2lstm_116/while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_116/while/lstm_cell_116/strided_slice_2/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   26
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_2StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_2:value:0;lstm_116/while/lstm_cell_116/strided_slice_2/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_2/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_2й
%lstm_116/while/lstm_cell_116/MatMul_6MatMul&lstm_116/while/lstm_cell_116/mul_2:z:05lstm_116/while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_6е
"lstm_116/while/lstm_cell_116/add_2AddV2/lstm_116/while/lstm_cell_116/BiasAdd_2:output:0/lstm_116/while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_2®
!lstm_116/while/lstm_cell_116/ReluRelu&lstm_116/while/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_116/while/lstm_cell_116/Relu№
"lstm_116/while/lstm_cell_116/mul_5Mul(lstm_116/while/lstm_cell_116/Sigmoid:y:0/lstm_116/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_5”
"lstm_116/while/lstm_cell_116/add_3AddV2&lstm_116/while/lstm_cell_116/mul_4:z:0&lstm_116/while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_3÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_3ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_3є
2lstm_116/while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_116/while/lstm_cell_116/strided_slice_3/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_3StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_3:value:0;lstm_116/while/lstm_cell_116/strided_slice_3/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_3/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_3й
%lstm_116/while/lstm_cell_116/MatMul_7MatMul&lstm_116/while/lstm_cell_116/mul_3:z:05lstm_116/while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_7е
"lstm_116/while/lstm_cell_116/add_4AddV2/lstm_116/while/lstm_cell_116/BiasAdd_3:output:0/lstm_116/while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_4µ
&lstm_116/while/lstm_cell_116/Sigmoid_2Sigmoid&lstm_116/while/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/Sigmoid_2ђ
#lstm_116/while/lstm_cell_116/Relu_1Relu&lstm_116/while/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_116/while/lstm_cell_116/Relu_1а
"lstm_116/while/lstm_cell_116/mul_6Mul*lstm_116/while/lstm_cell_116/Sigmoid_2:y:01lstm_116/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_6О
3lstm_116/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_116_while_placeholder_1lstm_116_while_placeholder&lstm_116/while/lstm_cell_116/mul_6:z:0*
_output_shapes
: *
element_dtype025
3lstm_116/while/TensorArrayV2Write/TensorListSetItemn
lstm_116/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_116/while/add/yН
lstm_116/while/addAddV2lstm_116_while_placeholderlstm_116/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_116/while/addr
lstm_116/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_116/while/add_1/y£
lstm_116/while/add_1AddV2*lstm_116_while_lstm_116_while_loop_counterlstm_116/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_116/while/add_1П
lstm_116/while/IdentityIdentitylstm_116/while/add_1:z:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/IdentityЂ
lstm_116/while/Identity_1Identity0lstm_116_while_lstm_116_while_maximum_iterations^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_1С
lstm_116/while/Identity_2Identitylstm_116/while/add:z:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_2Њ
lstm_116/while/Identity_3IdentityClstm_116/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_3≤
lstm_116/while/Identity_4Identity&lstm_116/while/lstm_cell_116/mul_6:z:0^lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/while/Identity_4≤
lstm_116/while/Identity_5Identity&lstm_116/while/lstm_cell_116/add_3:z:0^lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/while/Identity_5Ф
lstm_116/while/NoOpNoOp,^lstm_116/while/lstm_cell_116/ReadVariableOp.^lstm_116/while/lstm_cell_116/ReadVariableOp_1.^lstm_116/while/lstm_cell_116/ReadVariableOp_2.^lstm_116/while/lstm_cell_116/ReadVariableOp_32^lstm_116/while/lstm_cell_116/split/ReadVariableOp4^lstm_116/while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_116/while/NoOp";
lstm_116_while_identity lstm_116/while/Identity:output:0"?
lstm_116_while_identity_1"lstm_116/while/Identity_1:output:0"?
lstm_116_while_identity_2"lstm_116/while/Identity_2:output:0"?
lstm_116_while_identity_3"lstm_116/while/Identity_3:output:0"?
lstm_116_while_identity_4"lstm_116/while/Identity_4:output:0"?
lstm_116_while_identity_5"lstm_116/while/Identity_5:output:0"T
'lstm_116_while_lstm_116_strided_slice_1)lstm_116_while_lstm_116_strided_slice_1_0"n
4lstm_116_while_lstm_cell_116_readvariableop_resource6lstm_116_while_lstm_cell_116_readvariableop_resource_0"~
<lstm_116_while_lstm_cell_116_split_1_readvariableop_resource>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0"z
:lstm_116_while_lstm_cell_116_split_readvariableop_resource<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0"ћ
clstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensorelstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2Z
+lstm_116/while/lstm_cell_116/ReadVariableOp+lstm_116/while/lstm_cell_116/ReadVariableOp2^
-lstm_116/while/lstm_cell_116/ReadVariableOp_1-lstm_116/while/lstm_cell_116/ReadVariableOp_12^
-lstm_116/while/lstm_cell_116/ReadVariableOp_2-lstm_116/while/lstm_cell_116/ReadVariableOp_22^
-lstm_116/while/lstm_cell_116/ReadVariableOp_3-lstm_116/while/lstm_cell_116/ReadVariableOp_32f
1lstm_116/while/lstm_cell_116/split/ReadVariableOp1lstm_116/while/lstm_cell_116/split/ReadVariableOp2j
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
»“
і
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705521

inputs>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
:€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_like
lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout/ConstЈ
lstm_cell_116/dropout/MulMul lstm_cell_116/ones_like:output:0$lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/MulК
lstm_cell_116/dropout/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout/Shapeы
2lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2йбХ24
2lstm_cell_116/dropout/random_uniform/RandomUniformС
$lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2&
$lstm_cell_116/dropout/GreaterEqual/yц
"lstm_cell_116/dropout/GreaterEqualGreaterEqual;lstm_cell_116/dropout/random_uniform/RandomUniform:output:0-lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_116/dropout/GreaterEqual©
lstm_cell_116/dropout/CastCast&lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Cast≤
lstm_cell_116/dropout/Mul_1Mullstm_cell_116/dropout/Mul:z:0lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Mul_1Г
lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_1/Constљ
lstm_cell_116/dropout_1/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/MulО
lstm_cell_116/dropout_1/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_1/ShapeБ
4lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2фхї26
4lstm_cell_116/dropout_1/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_1/GreaterEqual/yю
$lstm_cell_116/dropout_1/GreaterEqualGreaterEqual=lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_1/GreaterEqualѓ
lstm_cell_116/dropout_1/CastCast(lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/CastЇ
lstm_cell_116/dropout_1/Mul_1Mullstm_cell_116/dropout_1/Mul:z:0 lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/Mul_1Г
lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_2/Constљ
lstm_cell_116/dropout_2/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/MulО
lstm_cell_116/dropout_2/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_2/ShapeБ
4lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЭЋТ26
4lstm_cell_116/dropout_2/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_2/GreaterEqual/yю
$lstm_cell_116/dropout_2/GreaterEqualGreaterEqual=lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_2/GreaterEqualѓ
lstm_cell_116/dropout_2/CastCast(lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/CastЇ
lstm_cell_116/dropout_2/Mul_1Mullstm_cell_116/dropout_2/Mul:z:0 lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/Mul_1Г
lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_3/Constљ
lstm_cell_116/dropout_3/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/MulО
lstm_cell_116/dropout_3/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_3/ShapeА
4lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Ј≈-26
4lstm_cell_116/dropout_3/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_3/GreaterEqual/yю
$lstm_cell_116/dropout_3/GreaterEqualGreaterEqual=lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_3/GreaterEqualѓ
lstm_cell_116/dropout_3/CastCast(lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/CastЇ
lstm_cell_116/dropout_3/Mul_1Mullstm_cell_116/dropout_3/Mul:z:0 lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/Mul_1А
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3Р
lstm_cell_116/mulMulzeros:output:0lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulЦ
lstm_cell_116/mul_1Mulzeros:output:0!lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Ц
lstm_cell_116/mul_2Mulzeros:output:0!lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Ц
lstm_cell_116/mul_3Mulzeros:output:0!lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3705356*
condR
while_cond_3705355*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm•
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¬µ
±	
while_body_3704806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeЛ
!while/lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2#
!while/lstm_cell_116/dropout/Constѕ
while/lstm_cell_116/dropout/MulMul&while/lstm_cell_116/ones_like:output:0*while/lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_116/dropout/MulЬ
!while/lstm_cell_116/dropout/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_116/dropout/ShapeН
8while/lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2вГљ2:
8while/lstm_cell_116/dropout/random_uniform/RandomUniformЭ
*while/lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2,
*while/lstm_cell_116/dropout/GreaterEqual/yО
(while/lstm_cell_116/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_116/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_116/dropout/GreaterEqualї
 while/lstm_cell_116/dropout/CastCast,while/lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_116/dropout/Cast 
!while/lstm_cell_116/dropout/Mul_1Mul#while/lstm_cell_116/dropout/Mul:z:0$while/lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout/Mul_1П
#while/lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_1/Const’
!while/lstm_cell_116/dropout_1/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_1/Mul†
#while/lstm_cell_116/dropout_1/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_1/ShapeУ
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Із¬2<
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_1/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_1/GreaterEqualЅ
"while/lstm_cell_116/dropout_1/CastCast.while/lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_1/Cast“
#while/lstm_cell_116/dropout_1/Mul_1Mul%while/lstm_cell_116/dropout_1/Mul:z:0&while/lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_1/Mul_1П
#while/lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_2/Const’
!while/lstm_cell_116/dropout_2/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_2/Mul†
#while/lstm_cell_116/dropout_2/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_2/ShapeТ
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2цЂ42<
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_2/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_2/GreaterEqualЅ
"while/lstm_cell_116/dropout_2/CastCast.while/lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_2/Cast“
#while/lstm_cell_116/dropout_2/Mul_1Mul%while/lstm_cell_116/dropout_2/Mul:z:0&while/lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_2/Mul_1П
#while/lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_3/Const’
!while/lstm_cell_116/dropout_3/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_3/Mul†
#while/lstm_cell_116/dropout_3/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_3/ShapeУ
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2¶™«2<
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_3/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_3/GreaterEqualЅ
"while/lstm_cell_116/dropout_3/CastCast.while/lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_3/Cast“
#while/lstm_cell_116/dropout_3/Mul_1Mul%while/lstm_cell_116/dropout_3/Mul:z:0&while/lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_3/Mul_1М
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3І
while/lstm_cell_116/mulMulwhile_placeholder_2%while/lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul≠
while/lstm_cell_116/mul_1Mulwhile_placeholder_2'while/lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1≠
while/lstm_cell_116/mul_2Mulwhile_placeholder_2'while/lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2≠
while/lstm_cell_116/mul_3Mulwhile_placeholder_2'while/lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ю“
ґ
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704971
inputs_0>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_like
lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout/ConstЈ
lstm_cell_116/dropout/MulMul lstm_cell_116/ones_like:output:0$lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/MulК
lstm_cell_116/dropout/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout/Shapeы
2lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Џ”°24
2lstm_cell_116/dropout/random_uniform/RandomUniformС
$lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2&
$lstm_cell_116/dropout/GreaterEqual/yц
"lstm_cell_116/dropout/GreaterEqualGreaterEqual;lstm_cell_116/dropout/random_uniform/RandomUniform:output:0-lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_116/dropout/GreaterEqual©
lstm_cell_116/dropout/CastCast&lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Cast≤
lstm_cell_116/dropout/Mul_1Mullstm_cell_116/dropout/Mul:z:0lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout/Mul_1Г
lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_1/Constљ
lstm_cell_116/dropout_1/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/MulО
lstm_cell_116/dropout_1/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_1/ShapeА
4lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2уЭ.26
4lstm_cell_116/dropout_1/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_1/GreaterEqual/yю
$lstm_cell_116/dropout_1/GreaterEqualGreaterEqual=lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_1/GreaterEqualѓ
lstm_cell_116/dropout_1/CastCast(lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/CastЇ
lstm_cell_116/dropout_1/Mul_1Mullstm_cell_116/dropout_1/Mul:z:0 lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_1/Mul_1Г
lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_2/Constљ
lstm_cell_116/dropout_2/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/MulО
lstm_cell_116/dropout_2/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_2/ShapeБ
4lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2∞ОП26
4lstm_cell_116/dropout_2/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_2/GreaterEqual/yю
$lstm_cell_116/dropout_2/GreaterEqualGreaterEqual=lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_2/GreaterEqualѓ
lstm_cell_116/dropout_2/CastCast(lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/CastЇ
lstm_cell_116/dropout_2/Mul_1Mullstm_cell_116/dropout_2/Mul:z:0 lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_2/Mul_1Г
lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_116/dropout_3/Constљ
lstm_cell_116/dropout_3/MulMul lstm_cell_116/ones_like:output:0&lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/MulО
lstm_cell_116/dropout_3/ShapeShape lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_116/dropout_3/ShapeБ
4lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ОЛј26
4lstm_cell_116/dropout_3/random_uniform/RandomUniformХ
&lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&lstm_cell_116/dropout_3/GreaterEqual/yю
$lstm_cell_116/dropout_3/GreaterEqualGreaterEqual=lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_cell_116/dropout_3/GreaterEqualѓ
lstm_cell_116/dropout_3/CastCast(lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/CastЇ
lstm_cell_116/dropout_3/Mul_1Mullstm_cell_116/dropout_3/Mul:z:0 lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/dropout_3/Mul_1А
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3Р
lstm_cell_116/mulMulzeros:output:0lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulЦ
lstm_cell_116/mul_1Mulzeros:output:0!lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Ц
lstm_cell_116/mul_2Mulzeros:output:0!lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Ц
lstm_cell_116/mul_3Mulzeros:output:0!lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3704806*
condR
while_cond_3704805*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Џ,
ј
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703584

inputs#
lstm_116_3703553:	А
lstm_116_3703555:	А#
lstm_116_3703557:	 А#
dense_140_3703560:  
dense_140_3703562: #
dense_141_3703565: 
dense_141_3703567:
identityИҐ!dense_140/StatefulPartitionedCallҐ!dense_141/StatefulPartitionedCallҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ lstm_116/StatefulPartitionedCallҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpЂ
 lstm_116/StatefulPartitionedCallStatefulPartitionedCallinputslstm_116_3703553lstm_116_3703555lstm_116_3703557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37035202"
 lstm_116/StatefulPartitionedCallњ
!dense_140/StatefulPartitionedCallStatefulPartitionedCall)lstm_116/StatefulPartitionedCall:output:0dense_140_3703560dense_140_3703562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_37031012#
!dense_140/StatefulPartitionedCallј
!dense_141/StatefulPartitionedCallStatefulPartitionedCall*dense_140/StatefulPartitionedCall:output:0dense_141_3703565dense_141_3703567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_37031232#
!dense_141/StatefulPartitionedCallГ
reshape_70/PartitionedCallPartitionedCall*dense_141/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_70_layer_call_and_return_conditional_losses_37031422
reshape_70/PartitionedCall‘
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_116_3703553*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul≤
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_141_3703567*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulВ
IdentityIdentity#reshape_70/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЃ
NoOpNoOp"^dense_140/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall1^dense_141/bias/Regularizer/Square/ReadVariableOp!^lstm_116/StatefulPartitionedCall@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2D
 lstm_116/StatefulPartitionedCall lstm_116/StatefulPartitionedCall2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ
»
while_cond_3704805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3704805___redundant_placeholder05
1while_while_cond_3704805___redundant_placeholder15
1while_while_cond_3704805___redundant_placeholder25
1while_while_cond_3704805___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Џ
»
while_cond_3704530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3704530___redundant_placeholder05
1while_while_cond_3704530___redundant_placeholder15
1while_while_cond_3704530___redundant_placeholder25
1while_while_cond_3704530___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
™Щ
м
lstm_116_while_body_3703875.
*lstm_116_while_lstm_116_while_loop_counter4
0lstm_116_while_lstm_116_while_maximum_iterations
lstm_116_while_placeholder 
lstm_116_while_placeholder_1 
lstm_116_while_placeholder_2 
lstm_116_while_placeholder_3-
)lstm_116_while_lstm_116_strided_slice_1_0i
elstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0:	АM
>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0:	АI
6lstm_116_while_lstm_cell_116_readvariableop_resource_0:	 А
lstm_116_while_identity
lstm_116_while_identity_1
lstm_116_while_identity_2
lstm_116_while_identity_3
lstm_116_while_identity_4
lstm_116_while_identity_5+
'lstm_116_while_lstm_116_strided_slice_1g
clstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensorM
:lstm_116_while_lstm_cell_116_split_readvariableop_resource:	АK
<lstm_116_while_lstm_cell_116_split_1_readvariableop_resource:	АG
4lstm_116_while_lstm_cell_116_readvariableop_resource:	 АИҐ+lstm_116/while/lstm_cell_116/ReadVariableOpҐ-lstm_116/while/lstm_cell_116/ReadVariableOp_1Ґ-lstm_116/while/lstm_cell_116/ReadVariableOp_2Ґ-lstm_116/while/lstm_cell_116/ReadVariableOp_3Ґ1lstm_116/while/lstm_cell_116/split/ReadVariableOpҐ3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp’
@lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2B
@lstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
2lstm_116/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0lstm_116_while_placeholderIlstm_116/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype024
2lstm_116/while/TensorArrayV2Read/TensorListGetItem®
,lstm_116/while/lstm_cell_116/ones_like/ShapeShapelstm_116_while_placeholder_2*
T0*
_output_shapes
:2.
,lstm_116/while/lstm_cell_116/ones_like/Shape°
,lstm_116/while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,lstm_116/while/lstm_cell_116/ones_like/Constш
&lstm_116/while/lstm_cell_116/ones_likeFill5lstm_116/while/lstm_cell_116/ones_like/Shape:output:05lstm_116/while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/ones_likeЮ
,lstm_116/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_116/while/lstm_cell_116/split/split_dimд
1lstm_116/while/lstm_cell_116/split/ReadVariableOpReadVariableOp<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype023
1lstm_116/while/lstm_cell_116/split/ReadVariableOpЫ
"lstm_116/while/lstm_cell_116/splitSplit5lstm_116/while/lstm_cell_116/split/split_dim:output:09lstm_116/while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2$
"lstm_116/while/lstm_cell_116/splitо
#lstm_116/while/lstm_cell_116/MatMulMatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_116/while/lstm_cell_116/MatMulт
%lstm_116/while/lstm_cell_116/MatMul_1MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_1т
%lstm_116/while/lstm_cell_116/MatMul_2MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_2т
%lstm_116/while/lstm_cell_116/MatMul_3MatMul9lstm_116/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_116/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_3Ґ
.lstm_116/while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.lstm_116/while/lstm_cell_116/split_1/split_dimж
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype025
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOpУ
$lstm_116/while/lstm_cell_116/split_1Split7lstm_116/while/lstm_cell_116/split_1/split_dim:output:0;lstm_116/while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2&
$lstm_116/while/lstm_cell_116/split_1з
$lstm_116/while/lstm_cell_116/BiasAddBiasAdd-lstm_116/while/lstm_cell_116/MatMul:product:0-lstm_116/while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/while/lstm_cell_116/BiasAddн
&lstm_116/while/lstm_cell_116/BiasAdd_1BiasAdd/lstm_116/while/lstm_cell_116/MatMul_1:product:0-lstm_116/while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_1н
&lstm_116/while/lstm_cell_116/BiasAdd_2BiasAdd/lstm_116/while/lstm_cell_116/MatMul_2:product:0-lstm_116/while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_2н
&lstm_116/while/lstm_cell_116/BiasAdd_3BiasAdd/lstm_116/while/lstm_cell_116/MatMul_3:product:0-lstm_116/while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/BiasAdd_3ћ
 lstm_116/while/lstm_cell_116/mulMullstm_116_while_placeholder_2/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/while/lstm_cell_116/mul–
"lstm_116/while/lstm_cell_116/mul_1Mullstm_116_while_placeholder_2/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_1–
"lstm_116/while/lstm_cell_116/mul_2Mullstm_116_while_placeholder_2/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_2–
"lstm_116/while/lstm_cell_116/mul_3Mullstm_116_while_placeholder_2/lstm_116/while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_3“
+lstm_116/while/lstm_cell_116/ReadVariableOpReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_116/while/lstm_cell_116/ReadVariableOpµ
0lstm_116/while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_116/while/lstm_cell_116/strided_slice/stackє
2lstm_116/while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_116/while/lstm_cell_116/strided_slice/stack_1є
2lstm_116/while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_116/while/lstm_cell_116/strided_slice/stack_2™
*lstm_116/while/lstm_cell_116/strided_sliceStridedSlice3lstm_116/while/lstm_cell_116/ReadVariableOp:value:09lstm_116/while/lstm_cell_116/strided_slice/stack:output:0;lstm_116/while/lstm_cell_116/strided_slice/stack_1:output:0;lstm_116/while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_116/while/lstm_cell_116/strided_sliceе
%lstm_116/while/lstm_cell_116/MatMul_4MatMul$lstm_116/while/lstm_cell_116/mul:z:03lstm_116/while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_4я
 lstm_116/while/lstm_cell_116/addAddV2-lstm_116/while/lstm_cell_116/BiasAdd:output:0/lstm_116/while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/while/lstm_cell_116/addѓ
$lstm_116/while/lstm_cell_116/SigmoidSigmoid$lstm_116/while/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/while/lstm_cell_116/Sigmoid÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_1ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_1є
2lstm_116/while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_116/while/lstm_cell_116/strided_slice_1/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   26
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_1/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_1StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_1:value:0;lstm_116/while/lstm_cell_116/strided_slice_1/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_1/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_1й
%lstm_116/while/lstm_cell_116/MatMul_5MatMul&lstm_116/while/lstm_cell_116/mul_1:z:05lstm_116/while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_5е
"lstm_116/while/lstm_cell_116/add_1AddV2/lstm_116/while/lstm_cell_116/BiasAdd_1:output:0/lstm_116/while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_1µ
&lstm_116/while/lstm_cell_116/Sigmoid_1Sigmoid&lstm_116/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/Sigmoid_1Ћ
"lstm_116/while/lstm_cell_116/mul_4Mul*lstm_116/while/lstm_cell_116/Sigmoid_1:y:0lstm_116_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_4÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_2ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_2є
2lstm_116/while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_116/while/lstm_cell_116/strided_slice_2/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   26
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_2/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_2StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_2:value:0;lstm_116/while/lstm_cell_116/strided_slice_2/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_2/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_2й
%lstm_116/while/lstm_cell_116/MatMul_6MatMul&lstm_116/while/lstm_cell_116/mul_2:z:05lstm_116/while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_6е
"lstm_116/while/lstm_cell_116/add_2AddV2/lstm_116/while/lstm_cell_116/BiasAdd_2:output:0/lstm_116/while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_2®
!lstm_116/while/lstm_cell_116/ReluRelu&lstm_116/while/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_116/while/lstm_cell_116/Relu№
"lstm_116/while/lstm_cell_116/mul_5Mul(lstm_116/while/lstm_cell_116/Sigmoid:y:0/lstm_116/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_5”
"lstm_116/while/lstm_cell_116/add_3AddV2&lstm_116/while/lstm_cell_116/mul_4:z:0&lstm_116/while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_3÷
-lstm_116/while/lstm_cell_116/ReadVariableOp_3ReadVariableOp6lstm_116_while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_116/while/lstm_cell_116/ReadVariableOp_3є
2lstm_116/while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_116/while/lstm_cell_116/strided_slice_3/stackљ
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_1љ
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_116/while/lstm_cell_116/strided_slice_3/stack_2ґ
,lstm_116/while/lstm_cell_116/strided_slice_3StridedSlice5lstm_116/while/lstm_cell_116/ReadVariableOp_3:value:0;lstm_116/while/lstm_cell_116/strided_slice_3/stack:output:0=lstm_116/while/lstm_cell_116/strided_slice_3/stack_1:output:0=lstm_116/while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_116/while/lstm_cell_116/strided_slice_3й
%lstm_116/while/lstm_cell_116/MatMul_7MatMul&lstm_116/while/lstm_cell_116/mul_3:z:05lstm_116/while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/while/lstm_cell_116/MatMul_7е
"lstm_116/while/lstm_cell_116/add_4AddV2/lstm_116/while/lstm_cell_116/BiasAdd_3:output:0/lstm_116/while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/add_4µ
&lstm_116/while/lstm_cell_116/Sigmoid_2Sigmoid&lstm_116/while/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/while/lstm_cell_116/Sigmoid_2ђ
#lstm_116/while/lstm_cell_116/Relu_1Relu&lstm_116/while/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_116/while/lstm_cell_116/Relu_1а
"lstm_116/while/lstm_cell_116/mul_6Mul*lstm_116/while/lstm_cell_116/Sigmoid_2:y:01lstm_116/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/while/lstm_cell_116/mul_6О
3lstm_116/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_116_while_placeholder_1lstm_116_while_placeholder&lstm_116/while/lstm_cell_116/mul_6:z:0*
_output_shapes
: *
element_dtype025
3lstm_116/while/TensorArrayV2Write/TensorListSetItemn
lstm_116/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_116/while/add/yН
lstm_116/while/addAddV2lstm_116_while_placeholderlstm_116/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_116/while/addr
lstm_116/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_116/while/add_1/y£
lstm_116/while/add_1AddV2*lstm_116_while_lstm_116_while_loop_counterlstm_116/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_116/while/add_1П
lstm_116/while/IdentityIdentitylstm_116/while/add_1:z:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/IdentityЂ
lstm_116/while/Identity_1Identity0lstm_116_while_lstm_116_while_maximum_iterations^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_1С
lstm_116/while/Identity_2Identitylstm_116/while/add:z:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_2Њ
lstm_116/while/Identity_3IdentityClstm_116/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_116/while/NoOp*
T0*
_output_shapes
: 2
lstm_116/while/Identity_3≤
lstm_116/while/Identity_4Identity&lstm_116/while/lstm_cell_116/mul_6:z:0^lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/while/Identity_4≤
lstm_116/while/Identity_5Identity&lstm_116/while/lstm_cell_116/add_3:z:0^lstm_116/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/while/Identity_5Ф
lstm_116/while/NoOpNoOp,^lstm_116/while/lstm_cell_116/ReadVariableOp.^lstm_116/while/lstm_cell_116/ReadVariableOp_1.^lstm_116/while/lstm_cell_116/ReadVariableOp_2.^lstm_116/while/lstm_cell_116/ReadVariableOp_32^lstm_116/while/lstm_cell_116/split/ReadVariableOp4^lstm_116/while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_116/while/NoOp";
lstm_116_while_identity lstm_116/while/Identity:output:0"?
lstm_116_while_identity_1"lstm_116/while/Identity_1:output:0"?
lstm_116_while_identity_2"lstm_116/while/Identity_2:output:0"?
lstm_116_while_identity_3"lstm_116/while/Identity_3:output:0"?
lstm_116_while_identity_4"lstm_116/while/Identity_4:output:0"?
lstm_116_while_identity_5"lstm_116/while/Identity_5:output:0"T
'lstm_116_while_lstm_116_strided_slice_1)lstm_116_while_lstm_116_strided_slice_1_0"n
4lstm_116_while_lstm_cell_116_readvariableop_resource6lstm_116_while_lstm_cell_116_readvariableop_resource_0"~
<lstm_116_while_lstm_cell_116_split_1_readvariableop_resource>lstm_116_while_lstm_cell_116_split_1_readvariableop_resource_0"z
:lstm_116_while_lstm_cell_116_split_readvariableop_resource<lstm_116_while_lstm_cell_116_split_readvariableop_resource_0"ћ
clstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensorelstm_116_while_tensorarrayv2read_tensorlistgetitem_lstm_116_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2Z
+lstm_116/while/lstm_cell_116/ReadVariableOp+lstm_116/while/lstm_cell_116/ReadVariableOp2^
-lstm_116/while/lstm_cell_116/ReadVariableOp_1-lstm_116/while/lstm_cell_116/ReadVariableOp_12^
-lstm_116/while/lstm_cell_116/ReadVariableOp_2-lstm_116/while/lstm_cell_116/ReadVariableOp_22^
-lstm_116/while/lstm_cell_116/ReadVariableOp_3-lstm_116/while/lstm_cell_116/ReadVariableOp_32f
1lstm_116/while/lstm_cell_116/split/ReadVariableOp1lstm_116/while/lstm_cell_116/split/ReadVariableOp2j
3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp3lstm_116/while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ЛВ
±	
while_body_3705081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeМ
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3®
while/lstm_cell_116/mulMulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mulђ
while/lstm_cell_116/mul_1Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1ђ
while/lstm_cell_116/mul_2Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2ђ
while/lstm_cell_116/mul_3Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Џ
»
while_cond_3705355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3705355___redundant_placeholder05
1while_while_cond_3705355___redundant_placeholder15
1while_while_cond_3705355___redundant_placeholder25
1while_while_cond_3705355___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Ї
ш
/__inference_lstm_cell_116_layer_call_fn_3705624

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37021732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/1
Д
ч
F__inference_dense_140_layer_call_and_return_conditional_losses_3703101

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
й

ь
lstm_116_while_cond_3704177.
*lstm_116_while_lstm_116_while_loop_counter4
0lstm_116_while_lstm_116_while_maximum_iterations
lstm_116_while_placeholder 
lstm_116_while_placeholder_1 
lstm_116_while_placeholder_2 
lstm_116_while_placeholder_30
,lstm_116_while_less_lstm_116_strided_slice_1G
Clstm_116_while_lstm_116_while_cond_3704177___redundant_placeholder0G
Clstm_116_while_lstm_116_while_cond_3704177___redundant_placeholder1G
Clstm_116_while_lstm_116_while_cond_3704177___redundant_placeholder2G
Clstm_116_while_lstm_116_while_cond_3704177___redundant_placeholder3
lstm_116_while_identity
Э
lstm_116/while/LessLesslstm_116_while_placeholder,lstm_116_while_less_lstm_116_strided_slice_1*
T0*
_output_shapes
: 2
lstm_116/while/Lessx
lstm_116/while/IdentityIdentitylstm_116/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_116/while/Identity";
lstm_116_while_identity lstm_116/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
√µ
±	
while_body_3705356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeЛ
!while/lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2#
!while/lstm_cell_116/dropout/Constѕ
while/lstm_cell_116/dropout/MulMul&while/lstm_cell_116/ones_like:output:0*while/lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_116/dropout/MulЬ
!while/lstm_cell_116/dropout/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_116/dropout/ShapeН
8while/lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2∞„щ2:
8while/lstm_cell_116/dropout/random_uniform/RandomUniformЭ
*while/lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2,
*while/lstm_cell_116/dropout/GreaterEqual/yО
(while/lstm_cell_116/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_116/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_116/dropout/GreaterEqualї
 while/lstm_cell_116/dropout/CastCast,while/lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_116/dropout/Cast 
!while/lstm_cell_116/dropout/Mul_1Mul#while/lstm_cell_116/dropout/Mul:z:0$while/lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout/Mul_1П
#while/lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_1/Const’
!while/lstm_cell_116/dropout_1/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_1/Mul†
#while/lstm_cell_116/dropout_1/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_1/ShapeУ
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2о†Ю2<
:while/lstm_cell_116/dropout_1/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_1/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_1/GreaterEqualЅ
"while/lstm_cell_116/dropout_1/CastCast.while/lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_1/Cast“
#while/lstm_cell_116/dropout_1/Mul_1Mul%while/lstm_cell_116/dropout_1/Mul:z:0&while/lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_1/Mul_1П
#while/lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_2/Const’
!while/lstm_cell_116/dropout_2/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_2/Mul†
#while/lstm_cell_116/dropout_2/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_2/ShapeУ
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ўЁ√2<
:while/lstm_cell_116/dropout_2/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_2/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_2/GreaterEqualЅ
"while/lstm_cell_116/dropout_2/CastCast.while/lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_2/Cast“
#while/lstm_cell_116/dropout_2/Mul_1Mul%while/lstm_cell_116/dropout_2/Mul:z:0&while/lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_2/Mul_1П
#while/lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2%
#while/lstm_cell_116/dropout_3/Const’
!while/lstm_cell_116/dropout_3/MulMul&while/lstm_cell_116/ones_like:output:0,while/lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_116/dropout_3/Mul†
#while/lstm_cell_116/dropout_3/ShapeShape&while/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_116/dropout_3/ShapeУ
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ѕЪэ2<
:while/lstm_cell_116/dropout_3/random_uniform/RandomUniform°
,while/lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2.
,while/lstm_cell_116/dropout_3/GreaterEqual/yЦ
*while/lstm_cell_116/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*while/lstm_cell_116/dropout_3/GreaterEqualЅ
"while/lstm_cell_116/dropout_3/CastCast.while/lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_116/dropout_3/Cast“
#while/lstm_cell_116/dropout_3/Mul_1Mul%while/lstm_cell_116/dropout_3/Mul:z:0&while/lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#while/lstm_cell_116/dropout_3/Mul_1М
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3І
while/lstm_cell_116/mulMulwhile_placeholder_2%while/lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul≠
while/lstm_cell_116/mul_1Mulwhile_placeholder_2'while/lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1≠
while/lstm_cell_116/mul_2Mulwhile_placeholder_2'while/lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2≠
while/lstm_cell_116/mul_3Mulwhile_placeholder_2'while/lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
†§
ґ
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704664
inputs_0>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_likeА
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3С
lstm_cell_116/mulMulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulХ
lstm_cell_116/mul_1Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Х
lstm_cell_116/mul_2Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Х
lstm_cell_116/mul_3Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3704531*
condR
while_cond_3704530*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
д	
®
/__inference_sequential_47_layer_call_fn_3703746

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_47_layer_call_and_return_conditional_losses_37031572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
д	
®
/__inference_sequential_47_layer_call_fn_3703765

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_47_layer_call_and_return_conditional_losses_37035842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
Ш
+__inference_dense_141_layer_call_fn_3705556

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_37031232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ЊЬ
Ї
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704371

inputsG
4lstm_116_lstm_cell_116_split_readvariableop_resource:	АE
6lstm_116_lstm_cell_116_split_1_readvariableop_resource:	АA
.lstm_116_lstm_cell_116_readvariableop_resource:	 А:
(dense_140_matmul_readvariableop_resource:  7
)dense_140_biasadd_readvariableop_resource: :
(dense_141_matmul_readvariableop_resource: 7
)dense_141_biasadd_readvariableop_resource:
identityИҐ dense_140/BiasAdd/ReadVariableOpҐdense_140/MatMul/ReadVariableOpҐ dense_141/BiasAdd/ReadVariableOpҐdense_141/MatMul/ReadVariableOpҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ%lstm_116/lstm_cell_116/ReadVariableOpҐ'lstm_116/lstm_cell_116/ReadVariableOp_1Ґ'lstm_116/lstm_cell_116/ReadVariableOp_2Ґ'lstm_116/lstm_cell_116/ReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐ+lstm_116/lstm_cell_116/split/ReadVariableOpҐ-lstm_116/lstm_cell_116/split_1/ReadVariableOpҐlstm_116/whileV
lstm_116/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_116/ShapeЖ
lstm_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_116/strided_slice/stackК
lstm_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_116/strided_slice/stack_1К
lstm_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_116/strided_slice/stack_2Ш
lstm_116/strided_sliceStridedSlicelstm_116/Shape:output:0%lstm_116/strided_slice/stack:output:0'lstm_116/strided_slice/stack_1:output:0'lstm_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_116/strided_slicen
lstm_116/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros/mul/yР
lstm_116/zeros/mulMullstm_116/strided_slice:output:0lstm_116/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros/mulq
lstm_116/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_116/zeros/Less/yЛ
lstm_116/zeros/LessLesslstm_116/zeros/mul:z:0lstm_116/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros/Lesst
lstm_116/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros/packed/1І
lstm_116/zeros/packedPacklstm_116/strided_slice:output:0 lstm_116/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_116/zeros/packedq
lstm_116/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/zeros/ConstЩ
lstm_116/zerosFilllstm_116/zeros/packed:output:0lstm_116/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/zerosr
lstm_116/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros_1/mul/yЦ
lstm_116/zeros_1/mulMullstm_116/strided_slice:output:0lstm_116/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros_1/mulu
lstm_116/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_116/zeros_1/Less/yУ
lstm_116/zeros_1/LessLesslstm_116/zeros_1/mul:z:0 lstm_116/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_116/zeros_1/Lessx
lstm_116/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/zeros_1/packed/1≠
lstm_116/zeros_1/packedPacklstm_116/strided_slice:output:0"lstm_116/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_116/zeros_1/packedu
lstm_116/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/zeros_1/Const°
lstm_116/zeros_1Fill lstm_116/zeros_1/packed:output:0lstm_116/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/zeros_1З
lstm_116/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_116/transpose/permХ
lstm_116/transpose	Transposeinputs lstm_116/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_116/transposej
lstm_116/Shape_1Shapelstm_116/transpose:y:0*
T0*
_output_shapes
:2
lstm_116/Shape_1К
lstm_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_116/strided_slice_1/stackО
 lstm_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_1/stack_1О
 lstm_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_1/stack_2§
lstm_116/strided_slice_1StridedSlicelstm_116/Shape_1:output:0'lstm_116/strided_slice_1/stack:output:0)lstm_116/strided_slice_1/stack_1:output:0)lstm_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_116/strided_slice_1Ч
$lstm_116/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2&
$lstm_116/TensorArrayV2/element_shape÷
lstm_116/TensorArrayV2TensorListReserve-lstm_116/TensorArrayV2/element_shape:output:0!lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_116/TensorArrayV2—
>lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2@
>lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shapeЬ
0lstm_116/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_116/transpose:y:0Glstm_116/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_116/TensorArrayUnstack/TensorListFromTensorК
lstm_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_116/strided_slice_2/stackО
 lstm_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_2/stack_1О
 lstm_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_2/stack_2≤
lstm_116/strided_slice_2StridedSlicelstm_116/transpose:y:0'lstm_116/strided_slice_2/stack:output:0)lstm_116/strided_slice_2/stack_1:output:0)lstm_116/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_116/strided_slice_2Ч
&lstm_116/lstm_cell_116/ones_like/ShapeShapelstm_116/zeros:output:0*
T0*
_output_shapes
:2(
&lstm_116/lstm_cell_116/ones_like/ShapeХ
&lstm_116/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2(
&lstm_116/lstm_cell_116/ones_like/Constа
 lstm_116/lstm_cell_116/ones_likeFill/lstm_116/lstm_cell_116/ones_like/Shape:output:0/lstm_116/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/ones_likeС
$lstm_116/lstm_cell_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2&
$lstm_116/lstm_cell_116/dropout/Constџ
"lstm_116/lstm_cell_116/dropout/MulMul)lstm_116/lstm_cell_116/ones_like:output:0-lstm_116/lstm_cell_116/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_116/lstm_cell_116/dropout/Mul•
$lstm_116/lstm_cell_116/dropout/ShapeShape)lstm_116/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_116/lstm_cell_116/dropout/ShapeЦ
;lstm_116/lstm_cell_116/dropout/random_uniform/RandomUniformRandomUniform-lstm_116/lstm_cell_116/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ƒµА2=
;lstm_116/lstm_cell_116/dropout/random_uniform/RandomUniform£
-lstm_116/lstm_cell_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_116/lstm_cell_116/dropout/GreaterEqual/yЪ
+lstm_116/lstm_cell_116/dropout/GreaterEqualGreaterEqualDlstm_116/lstm_cell_116/dropout/random_uniform/RandomUniform:output:06lstm_116/lstm_cell_116/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_116/lstm_cell_116/dropout/GreaterEqualƒ
#lstm_116/lstm_cell_116/dropout/CastCast/lstm_116/lstm_cell_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_116/lstm_cell_116/dropout/Cast÷
$lstm_116/lstm_cell_116/dropout/Mul_1Mul&lstm_116/lstm_cell_116/dropout/Mul:z:0'lstm_116/lstm_cell_116/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/lstm_cell_116/dropout/Mul_1Х
&lstm_116/lstm_cell_116/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2(
&lstm_116/lstm_cell_116/dropout_1/Constб
$lstm_116/lstm_cell_116/dropout_1/MulMul)lstm_116/lstm_cell_116/ones_like:output:0/lstm_116/lstm_cell_116/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/lstm_cell_116/dropout_1/Mul©
&lstm_116/lstm_cell_116/dropout_1/ShapeShape)lstm_116/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_116/lstm_cell_116/dropout_1/ShapeЫ
=lstm_116/lstm_cell_116/dropout_1/random_uniform/RandomUniformRandomUniform/lstm_116/lstm_cell_116/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2≥ј:2?
=lstm_116/lstm_cell_116/dropout_1/random_uniform/RandomUniformІ
/lstm_116/lstm_cell_116/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>21
/lstm_116/lstm_cell_116/dropout_1/GreaterEqual/yҐ
-lstm_116/lstm_cell_116/dropout_1/GreaterEqualGreaterEqualFlstm_116/lstm_cell_116/dropout_1/random_uniform/RandomUniform:output:08lstm_116/lstm_cell_116/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-lstm_116/lstm_cell_116/dropout_1/GreaterEqual 
%lstm_116/lstm_cell_116/dropout_1/CastCast1lstm_116/lstm_cell_116/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/lstm_cell_116/dropout_1/Castё
&lstm_116/lstm_cell_116/dropout_1/Mul_1Mul(lstm_116/lstm_cell_116/dropout_1/Mul:z:0)lstm_116/lstm_cell_116/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/lstm_cell_116/dropout_1/Mul_1Х
&lstm_116/lstm_cell_116/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2(
&lstm_116/lstm_cell_116/dropout_2/Constб
$lstm_116/lstm_cell_116/dropout_2/MulMul)lstm_116/lstm_cell_116/ones_like:output:0/lstm_116/lstm_cell_116/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/lstm_cell_116/dropout_2/Mul©
&lstm_116/lstm_cell_116/dropout_2/ShapeShape)lstm_116/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_116/lstm_cell_116/dropout_2/ShapeЬ
=lstm_116/lstm_cell_116/dropout_2/random_uniform/RandomUniformRandomUniform/lstm_116/lstm_cell_116/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2іЦ≠2?
=lstm_116/lstm_cell_116/dropout_2/random_uniform/RandomUniformІ
/lstm_116/lstm_cell_116/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>21
/lstm_116/lstm_cell_116/dropout_2/GreaterEqual/yҐ
-lstm_116/lstm_cell_116/dropout_2/GreaterEqualGreaterEqualFlstm_116/lstm_cell_116/dropout_2/random_uniform/RandomUniform:output:08lstm_116/lstm_cell_116/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-lstm_116/lstm_cell_116/dropout_2/GreaterEqual 
%lstm_116/lstm_cell_116/dropout_2/CastCast1lstm_116/lstm_cell_116/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/lstm_cell_116/dropout_2/Castё
&lstm_116/lstm_cell_116/dropout_2/Mul_1Mul(lstm_116/lstm_cell_116/dropout_2/Mul:z:0)lstm_116/lstm_cell_116/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/lstm_cell_116/dropout_2/Mul_1Х
&lstm_116/lstm_cell_116/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2(
&lstm_116/lstm_cell_116/dropout_3/Constб
$lstm_116/lstm_cell_116/dropout_3/MulMul)lstm_116/lstm_cell_116/ones_like:output:0/lstm_116/lstm_cell_116/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_116/lstm_cell_116/dropout_3/Mul©
&lstm_116/lstm_cell_116/dropout_3/ShapeShape)lstm_116/lstm_cell_116/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_116/lstm_cell_116/dropout_3/ShapeЬ
=lstm_116/lstm_cell_116/dropout_3/random_uniform/RandomUniformRandomUniform/lstm_116/lstm_cell_116/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Ћсъ2?
=lstm_116/lstm_cell_116/dropout_3/random_uniform/RandomUniformІ
/lstm_116/lstm_cell_116/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>21
/lstm_116/lstm_cell_116/dropout_3/GreaterEqual/yҐ
-lstm_116/lstm_cell_116/dropout_3/GreaterEqualGreaterEqualFlstm_116/lstm_cell_116/dropout_3/random_uniform/RandomUniform:output:08lstm_116/lstm_cell_116/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-lstm_116/lstm_cell_116/dropout_3/GreaterEqual 
%lstm_116/lstm_cell_116/dropout_3/CastCast1lstm_116/lstm_cell_116/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_116/lstm_cell_116/dropout_3/Castё
&lstm_116/lstm_cell_116/dropout_3/Mul_1Mul(lstm_116/lstm_cell_116/dropout_3/Mul:z:0)lstm_116/lstm_cell_116/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_116/lstm_cell_116/dropout_3/Mul_1Т
&lstm_116/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_116/lstm_cell_116/split/split_dim–
+lstm_116/lstm_cell_116/split/ReadVariableOpReadVariableOp4lstm_116_lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02-
+lstm_116/lstm_cell_116/split/ReadVariableOpГ
lstm_116/lstm_cell_116/splitSplit/lstm_116/lstm_cell_116/split/split_dim:output:03lstm_116/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_116/lstm_cell_116/splitƒ
lstm_116/lstm_cell_116/MatMulMatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/MatMul»
lstm_116/lstm_cell_116/MatMul_1MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_1»
lstm_116/lstm_cell_116/MatMul_2MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_2»
lstm_116/lstm_cell_116/MatMul_3MatMul!lstm_116/strided_slice_2:output:0%lstm_116/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_3Ц
(lstm_116/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_116/lstm_cell_116/split_1/split_dim“
-lstm_116/lstm_cell_116/split_1/ReadVariableOpReadVariableOp6lstm_116_lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-lstm_116/lstm_cell_116/split_1/ReadVariableOpы
lstm_116/lstm_cell_116/split_1Split1lstm_116/lstm_cell_116/split_1/split_dim:output:05lstm_116/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2 
lstm_116/lstm_cell_116/split_1ѕ
lstm_116/lstm_cell_116/BiasAddBiasAdd'lstm_116/lstm_cell_116/MatMul:product:0'lstm_116/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_116/lstm_cell_116/BiasAdd’
 lstm_116/lstm_cell_116/BiasAdd_1BiasAdd)lstm_116/lstm_cell_116/MatMul_1:product:0'lstm_116/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_1’
 lstm_116/lstm_cell_116/BiasAdd_2BiasAdd)lstm_116/lstm_cell_116/MatMul_2:product:0'lstm_116/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_2’
 lstm_116/lstm_cell_116/BiasAdd_3BiasAdd)lstm_116/lstm_cell_116/MatMul_3:product:0'lstm_116/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/BiasAdd_3і
lstm_116/lstm_cell_116/mulMullstm_116/zeros:output:0(lstm_116/lstm_cell_116/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mulЇ
lstm_116/lstm_cell_116/mul_1Mullstm_116/zeros:output:0*lstm_116/lstm_cell_116/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_1Ї
lstm_116/lstm_cell_116/mul_2Mullstm_116/zeros:output:0*lstm_116/lstm_cell_116/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_2Ї
lstm_116/lstm_cell_116/mul_3Mullstm_116/zeros:output:0*lstm_116/lstm_cell_116/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_3Њ
%lstm_116/lstm_cell_116/ReadVariableOpReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_116/lstm_cell_116/ReadVariableOp©
*lstm_116/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_116/lstm_cell_116/strided_slice/stack≠
,lstm_116/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_116/lstm_cell_116/strided_slice/stack_1≠
,lstm_116/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_116/lstm_cell_116/strided_slice/stack_2Ж
$lstm_116/lstm_cell_116/strided_sliceStridedSlice-lstm_116/lstm_cell_116/ReadVariableOp:value:03lstm_116/lstm_cell_116/strided_slice/stack:output:05lstm_116/lstm_cell_116/strided_slice/stack_1:output:05lstm_116/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_116/lstm_cell_116/strided_sliceЌ
lstm_116/lstm_cell_116/MatMul_4MatMullstm_116/lstm_cell_116/mul:z:0-lstm_116/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_4«
lstm_116/lstm_cell_116/addAddV2'lstm_116/lstm_cell_116/BiasAdd:output:0)lstm_116/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/addЭ
lstm_116/lstm_cell_116/SigmoidSigmoidlstm_116/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_116/lstm_cell_116/Sigmoid¬
'lstm_116/lstm_cell_116/ReadVariableOp_1ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_1≠
,lstm_116/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_116/lstm_cell_116/strided_slice_1/stack±
.lstm_116/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_116/lstm_cell_116/strided_slice_1/stack_1±
.lstm_116/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_1/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_1StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_1:value:05lstm_116/lstm_cell_116/strided_slice_1/stack:output:07lstm_116/lstm_cell_116/strided_slice_1/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_1—
lstm_116/lstm_cell_116/MatMul_5MatMul lstm_116/lstm_cell_116/mul_1:z:0/lstm_116/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_5Ќ
lstm_116/lstm_cell_116/add_1AddV2)lstm_116/lstm_cell_116/BiasAdd_1:output:0)lstm_116/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_1£
 lstm_116/lstm_cell_116/Sigmoid_1Sigmoid lstm_116/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/Sigmoid_1ґ
lstm_116/lstm_cell_116/mul_4Mul$lstm_116/lstm_cell_116/Sigmoid_1:y:0lstm_116/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_4¬
'lstm_116/lstm_cell_116/ReadVariableOp_2ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_2≠
,lstm_116/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_116/lstm_cell_116/strided_slice_2/stack±
.lstm_116/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_116/lstm_cell_116/strided_slice_2/stack_1±
.lstm_116/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_2/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_2StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_2:value:05lstm_116/lstm_cell_116/strided_slice_2/stack:output:07lstm_116/lstm_cell_116/strided_slice_2/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_2—
lstm_116/lstm_cell_116/MatMul_6MatMul lstm_116/lstm_cell_116/mul_2:z:0/lstm_116/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_6Ќ
lstm_116/lstm_cell_116/add_2AddV2)lstm_116/lstm_cell_116/BiasAdd_2:output:0)lstm_116/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_2Ц
lstm_116/lstm_cell_116/ReluRelu lstm_116/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/Reluƒ
lstm_116/lstm_cell_116/mul_5Mul"lstm_116/lstm_cell_116/Sigmoid:y:0)lstm_116/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_5ї
lstm_116/lstm_cell_116/add_3AddV2 lstm_116/lstm_cell_116/mul_4:z:0 lstm_116/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_3¬
'lstm_116/lstm_cell_116/ReadVariableOp_3ReadVariableOp.lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_116/lstm_cell_116/ReadVariableOp_3≠
,lstm_116/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_116/lstm_cell_116/strided_slice_3/stack±
.lstm_116/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_116/lstm_cell_116/strided_slice_3/stack_1±
.lstm_116/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_116/lstm_cell_116/strided_slice_3/stack_2Т
&lstm_116/lstm_cell_116/strided_slice_3StridedSlice/lstm_116/lstm_cell_116/ReadVariableOp_3:value:05lstm_116/lstm_cell_116/strided_slice_3/stack:output:07lstm_116/lstm_cell_116/strided_slice_3/stack_1:output:07lstm_116/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_116/lstm_cell_116/strided_slice_3—
lstm_116/lstm_cell_116/MatMul_7MatMul lstm_116/lstm_cell_116/mul_3:z:0/lstm_116/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_116/lstm_cell_116/MatMul_7Ќ
lstm_116/lstm_cell_116/add_4AddV2)lstm_116/lstm_cell_116/BiasAdd_3:output:0)lstm_116/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/add_4£
 lstm_116/lstm_cell_116/Sigmoid_2Sigmoid lstm_116/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_116/lstm_cell_116/Sigmoid_2Ъ
lstm_116/lstm_cell_116/Relu_1Relu lstm_116/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/Relu_1»
lstm_116/lstm_cell_116/mul_6Mul$lstm_116/lstm_cell_116/Sigmoid_2:y:0+lstm_116/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_116/lstm_cell_116/mul_6°
&lstm_116/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2(
&lstm_116/TensorArrayV2_1/element_shape№
lstm_116/TensorArrayV2_1TensorListReserve/lstm_116/TensorArrayV2_1/element_shape:output:0!lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_116/TensorArrayV2_1`
lstm_116/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/timeС
!lstm_116/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2#
!lstm_116/while/maximum_iterations|
lstm_116/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_116/while/loop_counterН
lstm_116/whileWhile$lstm_116/while/loop_counter:output:0*lstm_116/while/maximum_iterations:output:0lstm_116/time:output:0!lstm_116/TensorArrayV2_1:handle:0lstm_116/zeros:output:0lstm_116/zeros_1:output:0!lstm_116/strided_slice_1:output:0@lstm_116/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_116_lstm_cell_116_split_readvariableop_resource6lstm_116_lstm_cell_116_split_1_readvariableop_resource.lstm_116_lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_116_while_body_3704178*'
condR
lstm_116_while_cond_3704177*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_116/while«
9lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2;
9lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeМ
+lstm_116/TensorArrayV2Stack/TensorListStackTensorListStacklstm_116/while:output:3Blstm_116/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02-
+lstm_116/TensorArrayV2Stack/TensorListStackУ
lstm_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2 
lstm_116/strided_slice_3/stackО
 lstm_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_116/strided_slice_3/stack_1О
 lstm_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_116/strided_slice_3/stack_2–
lstm_116/strided_slice_3StridedSlice4lstm_116/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_116/strided_slice_3/stack:output:0)lstm_116/strided_slice_3/stack_1:output:0)lstm_116/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_116/strided_slice_3Л
lstm_116/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_116/transpose_1/perm…
lstm_116/transpose_1	Transpose4lstm_116/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_116/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_116/transpose_1x
lstm_116/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_116/runtimeЂ
dense_140/MatMul/ReadVariableOpReadVariableOp(dense_140_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense_140/MatMul/ReadVariableOpђ
dense_140/MatMulMatMul!lstm_116/strided_slice_3:output:0'dense_140/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/MatMul™
 dense_140/BiasAdd/ReadVariableOpReadVariableOp)dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_140/BiasAdd/ReadVariableOp©
dense_140/BiasAddBiasAdddense_140/MatMul:product:0(dense_140/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/BiasAddv
dense_140/ReluReludense_140/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_140/ReluЂ
dense_141/MatMul/ReadVariableOpReadVariableOp(dense_141_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense_141/MatMul/ReadVariableOpІ
dense_141/MatMulMatMuldense_140/Relu:activations:0'dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_141/MatMul™
 dense_141/BiasAdd/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_141/BiasAdd/ReadVariableOp©
dense_141/BiasAddBiasAdddense_141/MatMul:product:0(dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_141/BiasAddn
reshape_70/ShapeShapedense_141/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_70/ShapeК
reshape_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_70/strided_slice/stackО
 reshape_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_70/strided_slice/stack_1О
 reshape_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_70/strided_slice/stack_2§
reshape_70/strided_sliceStridedSlicereshape_70/Shape:output:0'reshape_70/strided_slice/stack:output:0)reshape_70/strided_slice/stack_1:output:0)reshape_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_70/strided_slicez
reshape_70/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_70/Reshape/shape/1z
reshape_70/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_70/Reshape/shape/2„
reshape_70/Reshape/shapePack!reshape_70/strided_slice:output:0#reshape_70/Reshape/shape/1:output:0#reshape_70/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_70/Reshape/shape®
reshape_70/ReshapeReshapedense_141/BiasAdd:output:0!reshape_70/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_70/Reshapeш
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_116_lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul 
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulz
IdentityIdentityreshape_70/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityв
NoOpNoOp!^dense_140/BiasAdd/ReadVariableOp ^dense_140/MatMul/ReadVariableOp!^dense_141/BiasAdd/ReadVariableOp ^dense_141/MatMul/ReadVariableOp1^dense_141/bias/Regularizer/Square/ReadVariableOp&^lstm_116/lstm_cell_116/ReadVariableOp(^lstm_116/lstm_cell_116/ReadVariableOp_1(^lstm_116/lstm_cell_116/ReadVariableOp_2(^lstm_116/lstm_cell_116/ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp,^lstm_116/lstm_cell_116/split/ReadVariableOp.^lstm_116/lstm_cell_116/split_1/ReadVariableOp^lstm_116/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_140/BiasAdd/ReadVariableOp dense_140/BiasAdd/ReadVariableOp2B
dense_140/MatMul/ReadVariableOpdense_140/MatMul/ReadVariableOp2D
 dense_141/BiasAdd/ReadVariableOp dense_141/BiasAdd/ReadVariableOp2B
dense_141/MatMul/ReadVariableOpdense_141/MatMul/ReadVariableOp2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2N
%lstm_116/lstm_cell_116/ReadVariableOp%lstm_116/lstm_cell_116/ReadVariableOp2R
'lstm_116/lstm_cell_116/ReadVariableOp_1'lstm_116/lstm_cell_116/ReadVariableOp_12R
'lstm_116/lstm_cell_116/ReadVariableOp_2'lstm_116/lstm_cell_116/ReadVariableOp_22R
'lstm_116/lstm_cell_116/ReadVariableOp_3'lstm_116/lstm_cell_116/ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2Z
+lstm_116/lstm_cell_116/split/ReadVariableOp+lstm_116/lstm_cell_116/split/ReadVariableOp2^
-lstm_116/lstm_cell_116/split_1/ReadVariableOp-lstm_116/lstm_cell_116/split_1/ReadVariableOp2 
lstm_116/whilelstm_116/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а,
¬
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703654
input_48#
lstm_116_3703623:	А
lstm_116_3703625:	А#
lstm_116_3703627:	 А#
dense_140_3703630:  
dense_140_3703632: #
dense_141_3703635: 
dense_141_3703637:
identityИҐ!dense_140/StatefulPartitionedCallҐ!dense_141/StatefulPartitionedCallҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ lstm_116/StatefulPartitionedCallҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp≠
 lstm_116/StatefulPartitionedCallStatefulPartitionedCallinput_48lstm_116_3703623lstm_116_3703625lstm_116_3703627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37030822"
 lstm_116/StatefulPartitionedCallњ
!dense_140/StatefulPartitionedCallStatefulPartitionedCall)lstm_116/StatefulPartitionedCall:output:0dense_140_3703630dense_140_3703632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_37031012#
!dense_140/StatefulPartitionedCallј
!dense_141/StatefulPartitionedCallStatefulPartitionedCall*dense_140/StatefulPartitionedCall:output:0dense_141_3703635dense_141_3703637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_37031232#
!dense_141/StatefulPartitionedCallГ
reshape_70/PartitionedCallPartitionedCall*dense_141/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_70_layer_call_and_return_conditional_losses_37031422
reshape_70/PartitionedCall‘
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_116_3703623*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul≤
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_141_3703637*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulВ
IdentityIdentity#reshape_70/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЃ
NoOpNoOp"^dense_140/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall1^dense_141/bias/Regularizer/Square/ReadVariableOp!^lstm_116/StatefulPartitionedCall@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2D
 lstm_116/StatefulPartitionedCall lstm_116/StatefulPartitionedCall2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
Д
ч
F__inference_dense_140_layer_call_and_return_conditional_losses_3705541

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Г
™
F__inference_dense_141_layer_call_and_return_conditional_losses_3705572

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ0dense_141/bias/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddј
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity≤
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_141/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
а,
¬
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703688
input_48#
lstm_116_3703657:	А
lstm_116_3703659:	А#
lstm_116_3703661:	 А#
dense_140_3703664:  
dense_140_3703666: #
dense_141_3703669: 
dense_141_3703671:
identityИҐ!dense_140/StatefulPartitionedCallҐ!dense_141/StatefulPartitionedCallҐ0dense_141/bias/Regularizer/Square/ReadVariableOpҐ lstm_116/StatefulPartitionedCallҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp≠
 lstm_116/StatefulPartitionedCallStatefulPartitionedCallinput_48lstm_116_3703657lstm_116_3703659lstm_116_3703661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37035202"
 lstm_116/StatefulPartitionedCallњ
!dense_140/StatefulPartitionedCallStatefulPartitionedCall)lstm_116/StatefulPartitionedCall:output:0dense_140_3703664dense_140_3703666*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_37031012#
!dense_140/StatefulPartitionedCallј
!dense_141/StatefulPartitionedCallStatefulPartitionedCall*dense_140/StatefulPartitionedCall:output:0dense_141_3703669dense_141_3703671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_37031232#
!dense_141/StatefulPartitionedCallГ
reshape_70/PartitionedCallPartitionedCall*dense_141/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_reshape_70_layer_call_and_return_conditional_losses_37031422
reshape_70/PartitionedCall‘
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_116_3703657*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/mul≤
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_141_3703671*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mulВ
IdentityIdentity#reshape_70/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЃ
NoOpNoOp"^dense_140/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall1^dense_141/bias/Regularizer/Square/ReadVariableOp!^lstm_116/StatefulPartitionedCall@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp2D
 lstm_116/StatefulPartitionedCall lstm_116/StatefulPartitionedCall2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
Ї
ш
/__inference_lstm_cell_116_layer_call_fn_3705641

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37024062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/1
ЛS
п
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705722

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
:€€€€€€€€€ 2
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
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
:€€€€€€€€€ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice/stack_2ь
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
:€€€€€€€€€ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
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
:€€€€€€€€€ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6б
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_2К
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/1
к£
і
E__inference_lstm_116_layer_call_and_return_conditional_losses_3703082

inputs>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
:€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_likeА
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3С
lstm_cell_116/mulMulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulХ
lstm_cell_116/mul_1Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Х
lstm_cell_116/mul_2Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Х
lstm_cell_116/mul_3Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3702949*
condR
while_cond_3702948*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm•
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
К
c
G__inference_reshape_70_layer_call_and_return_conditional_losses_3703142

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
strided_slice/stack_2в
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
Reshape/shape/2†
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ
»
while_cond_3703354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3703354___redundant_placeholder05
1while_while_cond_3703354___redundant_placeholder15
1while_while_cond_3703354___redundant_placeholder25
1while_while_cond_3703354___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ЙC
Б
 __inference__traced_save_3705953
file_prefix/
+savev2_dense_140_kernel_read_readvariableop-
)savev2_dense_140_bias_read_readvariableop/
+savev2_dense_141_kernel_read_readvariableop-
)savev2_dense_141_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_lstm_116_lstm_cell_116_kernel_read_readvariableopF
Bsavev2_lstm_116_lstm_cell_116_recurrent_kernel_read_readvariableop:
6savev2_lstm_116_lstm_cell_116_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_140_kernel_m_read_readvariableop4
0savev2_adam_dense_140_bias_m_read_readvariableop6
2savev2_adam_dense_141_kernel_m_read_readvariableop4
0savev2_adam_dense_141_bias_m_read_readvariableopC
?savev2_adam_lstm_116_lstm_cell_116_kernel_m_read_readvariableopM
Isavev2_adam_lstm_116_lstm_cell_116_recurrent_kernel_m_read_readvariableopA
=savev2_adam_lstm_116_lstm_cell_116_bias_m_read_readvariableop6
2savev2_adam_dense_140_kernel_v_read_readvariableop4
0savev2_adam_dense_140_bias_v_read_readvariableop6
2savev2_adam_dense_141_kernel_v_read_readvariableop4
0savev2_adam_dense_141_bias_v_read_readvariableopC
?savev2_adam_lstm_116_lstm_cell_116_kernel_v_read_readvariableopM
Isavev2_adam_lstm_116_lstm_cell_116_recurrent_kernel_v_read_readvariableopA
=savev2_adam_lstm_116_lstm_cell_116_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename–
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*в
valueЎB’B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¬
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_140_kernel_read_readvariableop)savev2_dense_140_bias_read_readvariableop+savev2_dense_141_kernel_read_readvariableop)savev2_dense_141_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_lstm_116_lstm_cell_116_kernel_read_readvariableopBsavev2_lstm_116_lstm_cell_116_recurrent_kernel_read_readvariableop6savev2_lstm_116_lstm_cell_116_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_140_kernel_m_read_readvariableop0savev2_adam_dense_140_bias_m_read_readvariableop2savev2_adam_dense_141_kernel_m_read_readvariableop0savev2_adam_dense_141_bias_m_read_readvariableop?savev2_adam_lstm_116_lstm_cell_116_kernel_m_read_readvariableopIsavev2_adam_lstm_116_lstm_cell_116_recurrent_kernel_m_read_readvariableop=savev2_adam_lstm_116_lstm_cell_116_bias_m_read_readvariableop2savev2_adam_dense_140_kernel_v_read_readvariableop0savev2_adam_dense_140_bias_v_read_readvariableop2savev2_adam_dense_141_kernel_v_read_readvariableop0savev2_adam_dense_141_bias_v_read_readvariableop?savev2_adam_lstm_116_lstm_cell_116_kernel_v_read_readvariableopIsavev2_adam_lstm_116_lstm_cell_116_recurrent_kernel_v_read_readvariableop=savev2_adam_lstm_116_lstm_cell_116_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*ё
_input_shapesћ
…: :  : : :: : : : : :	А:	 А:А: : :  : : ::	А:	 А:А:  : : ::	А:	 А:А: 2(
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
П
ђ
__inference_loss_fn_0_3705601G
9dense_141_bias_regularizer_square_readvariableop_resource:
identityИҐ0dense_141/bias/Regularizer/Square/ReadVariableOpЏ
0dense_141/bias/Regularizer/Square/ReadVariableOpReadVariableOp9dense_141_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_141/bias/Regularizer/Square/ReadVariableOpѓ
!dense_141/bias/Regularizer/SquareSquare8dense_141/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_141/bias/Regularizer/SquareО
 dense_141/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_141/bias/Regularizer/ConstЇ
dense_141/bias/Regularizer/SumSum%dense_141/bias/Regularizer/Square:y:0)dense_141/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/SumЙ
 dense_141/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_141/bias/Regularizer/mul/xЉ
dense_141/bias/Regularizer/mulMul)dense_141/bias/Regularizer/mul/x:output:0'dense_141/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_141/bias/Regularizer/mull
IdentityIdentity"dense_141/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityБ
NoOpNoOp1^dense_141/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_141/bias/Regularizer/Square/ReadVariableOp0dense_141/bias/Regularizer/Square/ReadVariableOp
к£
і
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705214

inputs>
+lstm_cell_116_split_readvariableop_resource:	А<
-lstm_cell_116_split_1_readvariableop_resource:	А8
%lstm_cell_116_readvariableop_resource:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_116/ReadVariableOpҐlstm_cell_116/ReadVariableOp_1Ґlstm_cell_116/ReadVariableOp_2Ґlstm_cell_116/ReadVariableOp_3Ґ"lstm_cell_116/split/ReadVariableOpҐ$lstm_cell_116/split_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
:€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2|
lstm_cell_116/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_116/ones_like/ShapeГ
lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_116/ones_like/ConstЉ
lstm_cell_116/ones_likeFill&lstm_cell_116/ones_like/Shape:output:0&lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/ones_likeА
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_116/split/split_dimµ
"lstm_cell_116/split/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02$
"lstm_cell_116/split/ReadVariableOpя
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0*lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_116/split†
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul§
lstm_cell_116/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_1§
lstm_cell_116/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_2§
lstm_cell_116/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_3Д
lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_116/split_1/split_dimЈ
$lstm_cell_116/split_1/ReadVariableOpReadVariableOp-lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$lstm_cell_116/split_1/ReadVariableOp„
lstm_cell_116/split_1Split(lstm_cell_116/split_1/split_dim:output:0,lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_116/split_1Ђ
lstm_cell_116/BiasAddBiasAddlstm_cell_116/MatMul:product:0lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd±
lstm_cell_116/BiasAdd_1BiasAdd lstm_cell_116/MatMul_1:product:0lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_1±
lstm_cell_116/BiasAdd_2BiasAdd lstm_cell_116/MatMul_2:product:0lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_2±
lstm_cell_116/BiasAdd_3BiasAdd lstm_cell_116/MatMul_3:product:0lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/BiasAdd_3С
lstm_cell_116/mulMulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mulХ
lstm_cell_116/mul_1Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_1Х
lstm_cell_116/mul_2Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_2Х
lstm_cell_116/mul_3Mulzeros:output:0 lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_3£
lstm_cell_116/ReadVariableOpReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_116/ReadVariableOpЧ
!lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_116/strided_slice/stackЫ
#lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice/stack_1Ы
#lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_116/strided_slice/stack_2–
lstm_cell_116/strided_sliceStridedSlice$lstm_cell_116/ReadVariableOp:value:0*lstm_cell_116/strided_slice/stack:output:0,lstm_cell_116/strided_slice/stack_1:output:0,lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice©
lstm_cell_116/MatMul_4MatMullstm_cell_116/mul:z:0$lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_4£
lstm_cell_116/addAddV2lstm_cell_116/BiasAdd:output:0 lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/addВ
lstm_cell_116/SigmoidSigmoidlstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/SigmoidІ
lstm_cell_116/ReadVariableOp_1ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_1Ы
#lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_116/strided_slice_1/stackЯ
%lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_116/strided_slice_1/stack_1Я
%lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_1/stack_2№
lstm_cell_116/strided_slice_1StridedSlice&lstm_cell_116/ReadVariableOp_1:value:0,lstm_cell_116/strided_slice_1/stack:output:0.lstm_cell_116/strided_slice_1/stack_1:output:0.lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_1≠
lstm_cell_116/MatMul_5MatMullstm_cell_116/mul_1:z:0&lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_5©
lstm_cell_116/add_1AddV2 lstm_cell_116/BiasAdd_1:output:0 lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_1И
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_1Т
lstm_cell_116/mul_4Mullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_4І
lstm_cell_116/ReadVariableOp_2ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_2Ы
#lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_116/strided_slice_2/stackЯ
%lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_116/strided_slice_2/stack_1Я
%lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_2/stack_2№
lstm_cell_116/strided_slice_2StridedSlice&lstm_cell_116/ReadVariableOp_2:value:0,lstm_cell_116/strided_slice_2/stack:output:0.lstm_cell_116/strided_slice_2/stack_1:output:0.lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_2≠
lstm_cell_116/MatMul_6MatMullstm_cell_116/mul_2:z:0&lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_6©
lstm_cell_116/add_2AddV2 lstm_cell_116/BiasAdd_2:output:0 lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_2{
lstm_cell_116/ReluRelulstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu†
lstm_cell_116/mul_5Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_5Ч
lstm_cell_116/add_3AddV2lstm_cell_116/mul_4:z:0lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_3І
lstm_cell_116/ReadVariableOp_3ReadVariableOp%lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype02 
lstm_cell_116/ReadVariableOp_3Ы
#lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_116/strided_slice_3/stackЯ
%lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_116/strided_slice_3/stack_1Я
%lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_116/strided_slice_3/stack_2№
lstm_cell_116/strided_slice_3StridedSlice&lstm_cell_116/ReadVariableOp_3:value:0,lstm_cell_116/strided_slice_3/stack:output:0.lstm_cell_116/strided_slice_3/stack_1:output:0.lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_116/strided_slice_3≠
lstm_cell_116/MatMul_7MatMullstm_cell_116/mul_3:z:0&lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/MatMul_7©
lstm_cell_116/add_4AddV2 lstm_cell_116/BiasAdd_3:output:0 lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/add_4И
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Sigmoid_2
lstm_cell_116/Relu_1Relulstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/Relu_1§
lstm_cell_116/mul_6Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_116/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_116_split_readvariableop_resource-lstm_cell_116_split_1_readvariableop_resource%lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3705081*
condR
while_cond_3705080*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm•
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeп
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityж
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_116/ReadVariableOp^lstm_cell_116/ReadVariableOp_1^lstm_cell_116/ReadVariableOp_2^lstm_cell_116/ReadVariableOp_3#^lstm_cell_116/split/ReadVariableOp%^lstm_cell_116/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_116/ReadVariableOplstm_cell_116/ReadVariableOp2@
lstm_cell_116/ReadVariableOp_1lstm_cell_116/ReadVariableOp_12@
lstm_cell_116/ReadVariableOp_2lstm_cell_116/ReadVariableOp_22@
lstm_cell_116/ReadVariableOp_3lstm_cell_116/ReadVariableOp_32H
"lstm_cell_116/split/ReadVariableOp"lstm_cell_116/split/ReadVariableOp2L
$lstm_cell_116/split_1/ReadVariableOp$lstm_cell_116/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√Г
Ы	
"__inference__wrapped_model_3702049
input_48U
Bsequential_47_lstm_116_lstm_cell_116_split_readvariableop_resource:	АS
Dsequential_47_lstm_116_lstm_cell_116_split_1_readvariableop_resource:	АO
<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource:	 АH
6sequential_47_dense_140_matmul_readvariableop_resource:  E
7sequential_47_dense_140_biasadd_readvariableop_resource: H
6sequential_47_dense_141_matmul_readvariableop_resource: E
7sequential_47_dense_141_biasadd_readvariableop_resource:
identityИҐ.sequential_47/dense_140/BiasAdd/ReadVariableOpҐ-sequential_47/dense_140/MatMul/ReadVariableOpҐ.sequential_47/dense_141/BiasAdd/ReadVariableOpҐ-sequential_47/dense_141/MatMul/ReadVariableOpҐ3sequential_47/lstm_116/lstm_cell_116/ReadVariableOpҐ5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_1Ґ5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_2Ґ5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_3Ґ9sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOpҐ;sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOpҐsequential_47/lstm_116/whilet
sequential_47/lstm_116/ShapeShapeinput_48*
T0*
_output_shapes
:2
sequential_47/lstm_116/ShapeҐ
*sequential_47/lstm_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_47/lstm_116/strided_slice/stack¶
,sequential_47/lstm_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_47/lstm_116/strided_slice/stack_1¶
,sequential_47/lstm_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_47/lstm_116/strided_slice/stack_2м
$sequential_47/lstm_116/strided_sliceStridedSlice%sequential_47/lstm_116/Shape:output:03sequential_47/lstm_116/strided_slice/stack:output:05sequential_47/lstm_116/strided_slice/stack_1:output:05sequential_47/lstm_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_47/lstm_116/strided_sliceК
"sequential_47/lstm_116/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_47/lstm_116/zeros/mul/y»
 sequential_47/lstm_116/zeros/mulMul-sequential_47/lstm_116/strided_slice:output:0+sequential_47/lstm_116/zeros/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_47/lstm_116/zeros/mulН
#sequential_47/lstm_116/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2%
#sequential_47/lstm_116/zeros/Less/y√
!sequential_47/lstm_116/zeros/LessLess$sequential_47/lstm_116/zeros/mul:z:0,sequential_47/lstm_116/zeros/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_47/lstm_116/zeros/LessР
%sequential_47/lstm_116/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_47/lstm_116/zeros/packed/1я
#sequential_47/lstm_116/zeros/packedPack-sequential_47/lstm_116/strided_slice:output:0.sequential_47/lstm_116/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_47/lstm_116/zeros/packedН
"sequential_47/lstm_116/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_47/lstm_116/zeros/Const—
sequential_47/lstm_116/zerosFill,sequential_47/lstm_116/zeros/packed:output:0+sequential_47/lstm_116/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_47/lstm_116/zerosО
$sequential_47/lstm_116/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_47/lstm_116/zeros_1/mul/yќ
"sequential_47/lstm_116/zeros_1/mulMul-sequential_47/lstm_116/strided_slice:output:0-sequential_47/lstm_116/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2$
"sequential_47/lstm_116/zeros_1/mulС
%sequential_47/lstm_116/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2'
%sequential_47/lstm_116/zeros_1/Less/yЋ
#sequential_47/lstm_116/zeros_1/LessLess&sequential_47/lstm_116/zeros_1/mul:z:0.sequential_47/lstm_116/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2%
#sequential_47/lstm_116/zeros_1/LessФ
'sequential_47/lstm_116/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_47/lstm_116/zeros_1/packed/1е
%sequential_47/lstm_116/zeros_1/packedPack-sequential_47/lstm_116/strided_slice:output:00sequential_47/lstm_116/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_47/lstm_116/zeros_1/packedС
$sequential_47/lstm_116/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$sequential_47/lstm_116/zeros_1/Constў
sequential_47/lstm_116/zeros_1Fill.sequential_47/lstm_116/zeros_1/packed:output:0-sequential_47/lstm_116/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
sequential_47/lstm_116/zeros_1£
%sequential_47/lstm_116/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_47/lstm_116/transpose/permЅ
 sequential_47/lstm_116/transpose	Transposeinput_48.sequential_47/lstm_116/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"
 sequential_47/lstm_116/transposeФ
sequential_47/lstm_116/Shape_1Shape$sequential_47/lstm_116/transpose:y:0*
T0*
_output_shapes
:2 
sequential_47/lstm_116/Shape_1¶
,sequential_47/lstm_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_47/lstm_116/strided_slice_1/stack™
.sequential_47/lstm_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/lstm_116/strided_slice_1/stack_1™
.sequential_47/lstm_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/lstm_116/strided_slice_1/stack_2ш
&sequential_47/lstm_116/strided_slice_1StridedSlice'sequential_47/lstm_116/Shape_1:output:05sequential_47/lstm_116/strided_slice_1/stack:output:07sequential_47/lstm_116/strided_slice_1/stack_1:output:07sequential_47/lstm_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_47/lstm_116/strided_slice_1≥
2sequential_47/lstm_116/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€24
2sequential_47/lstm_116/TensorArrayV2/element_shapeО
$sequential_47/lstm_116/TensorArrayV2TensorListReserve;sequential_47/lstm_116/TensorArrayV2/element_shape:output:0/sequential_47/lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_47/lstm_116/TensorArrayV2н
Lsequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2N
Lsequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shape‘
>sequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_47/lstm_116/transpose:y:0Usequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensor¶
,sequential_47/lstm_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_47/lstm_116/strided_slice_2/stack™
.sequential_47/lstm_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/lstm_116/strided_slice_2/stack_1™
.sequential_47/lstm_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/lstm_116/strided_slice_2/stack_2Ж
&sequential_47/lstm_116/strided_slice_2StridedSlice$sequential_47/lstm_116/transpose:y:05sequential_47/lstm_116/strided_slice_2/stack:output:07sequential_47/lstm_116/strided_slice_2/stack_1:output:07sequential_47/lstm_116/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2(
&sequential_47/lstm_116/strided_slice_2Ѕ
4sequential_47/lstm_116/lstm_cell_116/ones_like/ShapeShape%sequential_47/lstm_116/zeros:output:0*
T0*
_output_shapes
:26
4sequential_47/lstm_116/lstm_cell_116/ones_like/Shape±
4sequential_47/lstm_116/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4sequential_47/lstm_116/lstm_cell_116/ones_like/ConstШ
.sequential_47/lstm_116/lstm_cell_116/ones_likeFill=sequential_47/lstm_116/lstm_cell_116/ones_like/Shape:output:0=sequential_47/lstm_116/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/ones_likeЃ
4sequential_47/lstm_116/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_47/lstm_116/lstm_cell_116/split/split_dimъ
9sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOpReadVariableOpBsequential_47_lstm_116_lstm_cell_116_split_readvariableop_resource*
_output_shapes
:	А*
dtype02;
9sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOpї
*sequential_47/lstm_116/lstm_cell_116/splitSplit=sequential_47/lstm_116/lstm_cell_116/split/split_dim:output:0Asequential_47/lstm_116/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2,
*sequential_47/lstm_116/lstm_cell_116/splitь
+sequential_47/lstm_116/lstm_cell_116/MatMulMatMul/sequential_47/lstm_116/strided_slice_2:output:03sequential_47/lstm_116/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_47/lstm_116/lstm_cell_116/MatMulА
-sequential_47/lstm_116/lstm_cell_116/MatMul_1MatMul/sequential_47/lstm_116/strided_slice_2:output:03sequential_47/lstm_116/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_1А
-sequential_47/lstm_116/lstm_cell_116/MatMul_2MatMul/sequential_47/lstm_116/strided_slice_2:output:03sequential_47/lstm_116/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_2А
-sequential_47/lstm_116/lstm_cell_116/MatMul_3MatMul/sequential_47/lstm_116/strided_slice_2:output:03sequential_47/lstm_116/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_3≤
6sequential_47/lstm_116/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_47/lstm_116/lstm_cell_116/split_1/split_dimь
;sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOpReadVariableOpDsequential_47_lstm_116_lstm_cell_116_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOp≥
,sequential_47/lstm_116/lstm_cell_116/split_1Split?sequential_47/lstm_116/lstm_cell_116/split_1/split_dim:output:0Csequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2.
,sequential_47/lstm_116/lstm_cell_116/split_1З
,sequential_47/lstm_116/lstm_cell_116/BiasAddBiasAdd5sequential_47/lstm_116/lstm_cell_116/MatMul:product:05sequential_47/lstm_116/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_47/lstm_116/lstm_cell_116/BiasAddН
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_1BiasAdd7sequential_47/lstm_116/lstm_cell_116/MatMul_1:product:05sequential_47/lstm_116/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_1Н
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_2BiasAdd7sequential_47/lstm_116/lstm_cell_116/MatMul_2:product:05sequential_47/lstm_116/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_2Н
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_3BiasAdd7sequential_47/lstm_116/lstm_cell_116/MatMul_3:product:05sequential_47/lstm_116/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/BiasAdd_3н
(sequential_47/lstm_116/lstm_cell_116/mulMul%sequential_47/lstm_116/zeros:output:07sequential_47/lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_47/lstm_116/lstm_cell_116/mulс
*sequential_47/lstm_116/lstm_cell_116/mul_1Mul%sequential_47/lstm_116/zeros:output:07sequential_47/lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_1с
*sequential_47/lstm_116/lstm_cell_116/mul_2Mul%sequential_47/lstm_116/zeros:output:07sequential_47/lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_2с
*sequential_47/lstm_116/lstm_cell_116/mul_3Mul%sequential_47/lstm_116/zeros:output:07sequential_47/lstm_116/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_3и
3sequential_47/lstm_116/lstm_cell_116/ReadVariableOpReadVariableOp<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype025
3sequential_47/lstm_116/lstm_cell_116/ReadVariableOp≈
8sequential_47/lstm_116/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_47/lstm_116/lstm_cell_116/strided_slice/stack…
:sequential_47/lstm_116/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_47/lstm_116/lstm_cell_116/strided_slice/stack_1…
:sequential_47/lstm_116/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_47/lstm_116/lstm_cell_116/strided_slice/stack_2Џ
2sequential_47/lstm_116/lstm_cell_116/strided_sliceStridedSlice;sequential_47/lstm_116/lstm_cell_116/ReadVariableOp:value:0Asequential_47/lstm_116/lstm_cell_116/strided_slice/stack:output:0Csequential_47/lstm_116/lstm_cell_116/strided_slice/stack_1:output:0Csequential_47/lstm_116/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_47/lstm_116/lstm_cell_116/strided_sliceЕ
-sequential_47/lstm_116/lstm_cell_116/MatMul_4MatMul,sequential_47/lstm_116/lstm_cell_116/mul:z:0;sequential_47/lstm_116/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_4€
(sequential_47/lstm_116/lstm_cell_116/addAddV25sequential_47/lstm_116/lstm_cell_116/BiasAdd:output:07sequential_47/lstm_116/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_47/lstm_116/lstm_cell_116/add«
,sequential_47/lstm_116/lstm_cell_116/SigmoidSigmoid,sequential_47/lstm_116/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_47/lstm_116/lstm_cell_116/Sigmoidм
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_1ReadVariableOp<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype027
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_1…
:sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stackЌ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_1Ќ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_2ж
4sequential_47/lstm_116/lstm_cell_116/strided_slice_1StridedSlice=sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_1:value:0Csequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_1:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_47/lstm_116/lstm_cell_116/strided_slice_1Й
-sequential_47/lstm_116/lstm_cell_116/MatMul_5MatMul.sequential_47/lstm_116/lstm_cell_116/mul_1:z:0=sequential_47/lstm_116/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_5Е
*sequential_47/lstm_116/lstm_cell_116/add_1AddV27sequential_47/lstm_116/lstm_cell_116/BiasAdd_1:output:07sequential_47/lstm_116/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/add_1Ќ
.sequential_47/lstm_116/lstm_cell_116/Sigmoid_1Sigmoid.sequential_47/lstm_116/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/Sigmoid_1о
*sequential_47/lstm_116/lstm_cell_116/mul_4Mul2sequential_47/lstm_116/lstm_cell_116/Sigmoid_1:y:0'sequential_47/lstm_116/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_4м
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_2ReadVariableOp<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype027
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_2…
:sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2<
:sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stackЌ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_1Ќ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_2ж
4sequential_47/lstm_116/lstm_cell_116/strided_slice_2StridedSlice=sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_2:value:0Csequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_1:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_47/lstm_116/lstm_cell_116/strided_slice_2Й
-sequential_47/lstm_116/lstm_cell_116/MatMul_6MatMul.sequential_47/lstm_116/lstm_cell_116/mul_2:z:0=sequential_47/lstm_116/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_6Е
*sequential_47/lstm_116/lstm_cell_116/add_2AddV27sequential_47/lstm_116/lstm_cell_116/BiasAdd_2:output:07sequential_47/lstm_116/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/add_2ј
)sequential_47/lstm_116/lstm_cell_116/ReluRelu.sequential_47/lstm_116/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_47/lstm_116/lstm_cell_116/Reluь
*sequential_47/lstm_116/lstm_cell_116/mul_5Mul0sequential_47/lstm_116/lstm_cell_116/Sigmoid:y:07sequential_47/lstm_116/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_5у
*sequential_47/lstm_116/lstm_cell_116/add_3AddV2.sequential_47/lstm_116/lstm_cell_116/mul_4:z:0.sequential_47/lstm_116/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/add_3м
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_3ReadVariableOp<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource*
_output_shapes
:	 А*
dtype027
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_3…
:sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2<
:sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stackЌ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_1Ќ
<sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_2ж
4sequential_47/lstm_116/lstm_cell_116/strided_slice_3StridedSlice=sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_3:value:0Csequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_1:output:0Esequential_47/lstm_116/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_47/lstm_116/lstm_cell_116/strided_slice_3Й
-sequential_47/lstm_116/lstm_cell_116/MatMul_7MatMul.sequential_47/lstm_116/lstm_cell_116/mul_3:z:0=sequential_47/lstm_116/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_47/lstm_116/lstm_cell_116/MatMul_7Е
*sequential_47/lstm_116/lstm_cell_116/add_4AddV27sequential_47/lstm_116/lstm_cell_116/BiasAdd_3:output:07sequential_47/lstm_116/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/add_4Ќ
.sequential_47/lstm_116/lstm_cell_116/Sigmoid_2Sigmoid.sequential_47/lstm_116/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_47/lstm_116/lstm_cell_116/Sigmoid_2ƒ
+sequential_47/lstm_116/lstm_cell_116/Relu_1Relu.sequential_47/lstm_116/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_47/lstm_116/lstm_cell_116/Relu_1А
*sequential_47/lstm_116/lstm_cell_116/mul_6Mul2sequential_47/lstm_116/lstm_cell_116/Sigmoid_2:y:09sequential_47/lstm_116/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_47/lstm_116/lstm_cell_116/mul_6љ
4sequential_47/lstm_116/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    26
4sequential_47/lstm_116/TensorArrayV2_1/element_shapeФ
&sequential_47/lstm_116/TensorArrayV2_1TensorListReserve=sequential_47/lstm_116/TensorArrayV2_1/element_shape:output:0/sequential_47/lstm_116/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&sequential_47/lstm_116/TensorArrayV2_1|
sequential_47/lstm_116/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_47/lstm_116/time≠
/sequential_47/lstm_116/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_47/lstm_116/while/maximum_iterationsШ
)sequential_47/lstm_116/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_47/lstm_116/while/loop_counterя
sequential_47/lstm_116/whileWhile2sequential_47/lstm_116/while/loop_counter:output:08sequential_47/lstm_116/while/maximum_iterations:output:0$sequential_47/lstm_116/time:output:0/sequential_47/lstm_116/TensorArrayV2_1:handle:0%sequential_47/lstm_116/zeros:output:0'sequential_47/lstm_116/zeros_1:output:0/sequential_47/lstm_116/strided_slice_1:output:0Nsequential_47/lstm_116/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bsequential_47_lstm_116_lstm_cell_116_split_readvariableop_resourceDsequential_47_lstm_116_lstm_cell_116_split_1_readvariableop_resource<sequential_47_lstm_116_lstm_cell_116_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_47_lstm_116_while_body_3701900*5
cond-R+
)sequential_47_lstm_116_while_cond_3701899*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
sequential_47/lstm_116/whileг
Gsequential_47/lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2I
Gsequential_47/lstm_116/TensorArrayV2Stack/TensorListStack/element_shapeƒ
9sequential_47/lstm_116/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_47/lstm_116/while:output:3Psequential_47/lstm_116/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02;
9sequential_47/lstm_116/TensorArrayV2Stack/TensorListStackѓ
,sequential_47/lstm_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2.
,sequential_47/lstm_116/strided_slice_3/stack™
.sequential_47/lstm_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_47/lstm_116/strided_slice_3/stack_1™
.sequential_47/lstm_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/lstm_116/strided_slice_3/stack_2§
&sequential_47/lstm_116/strided_slice_3StridedSliceBsequential_47/lstm_116/TensorArrayV2Stack/TensorListStack:tensor:05sequential_47/lstm_116/strided_slice_3/stack:output:07sequential_47/lstm_116/strided_slice_3/stack_1:output:07sequential_47/lstm_116/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2(
&sequential_47/lstm_116/strided_slice_3І
'sequential_47/lstm_116/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'sequential_47/lstm_116/transpose_1/permБ
"sequential_47/lstm_116/transpose_1	TransposeBsequential_47/lstm_116/TensorArrayV2Stack/TensorListStack:tensor:00sequential_47/lstm_116/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2$
"sequential_47/lstm_116/transpose_1Ф
sequential_47/lstm_116/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_47/lstm_116/runtime’
-sequential_47/dense_140/MatMul/ReadVariableOpReadVariableOp6sequential_47_dense_140_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_47/dense_140/MatMul/ReadVariableOpд
sequential_47/dense_140/MatMulMatMul/sequential_47/lstm_116/strided_slice_3:output:05sequential_47/dense_140/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
sequential_47/dense_140/MatMul‘
.sequential_47/dense_140/BiasAdd/ReadVariableOpReadVariableOp7sequential_47_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_47/dense_140/BiasAdd/ReadVariableOpб
sequential_47/dense_140/BiasAddBiasAdd(sequential_47/dense_140/MatMul:product:06sequential_47/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential_47/dense_140/BiasAdd†
sequential_47/dense_140/ReluRelu(sequential_47/dense_140/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_47/dense_140/Relu’
-sequential_47/dense_141/MatMul/ReadVariableOpReadVariableOp6sequential_47_dense_141_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-sequential_47/dense_141/MatMul/ReadVariableOpя
sequential_47/dense_141/MatMulMatMul*sequential_47/dense_140/Relu:activations:05sequential_47/dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2 
sequential_47/dense_141/MatMul‘
.sequential_47/dense_141/BiasAdd/ReadVariableOpReadVariableOp7sequential_47_dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_47/dense_141/BiasAdd/ReadVariableOpб
sequential_47/dense_141/BiasAddBiasAdd(sequential_47/dense_141/MatMul:product:06sequential_47/dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential_47/dense_141/BiasAddШ
sequential_47/reshape_70/ShapeShape(sequential_47/dense_141/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_47/reshape_70/Shape¶
,sequential_47/reshape_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_47/reshape_70/strided_slice/stack™
.sequential_47/reshape_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/reshape_70/strided_slice/stack_1™
.sequential_47/reshape_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_47/reshape_70/strided_slice/stack_2ш
&sequential_47/reshape_70/strided_sliceStridedSlice'sequential_47/reshape_70/Shape:output:05sequential_47/reshape_70/strided_slice/stack:output:07sequential_47/reshape_70/strided_slice/stack_1:output:07sequential_47/reshape_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_47/reshape_70/strided_sliceЦ
(sequential_47/reshape_70/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_47/reshape_70/Reshape/shape/1Ц
(sequential_47/reshape_70/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_47/reshape_70/Reshape/shape/2Э
&sequential_47/reshape_70/Reshape/shapePack/sequential_47/reshape_70/strided_slice:output:01sequential_47/reshape_70/Reshape/shape/1:output:01sequential_47/reshape_70/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_47/reshape_70/Reshape/shapeа
 sequential_47/reshape_70/ReshapeReshape(sequential_47/dense_141/BiasAdd:output:0/sequential_47/reshape_70/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"
 sequential_47/reshape_70/ReshapeИ
IdentityIdentity)sequential_47/reshape_70/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЗ
NoOpNoOp/^sequential_47/dense_140/BiasAdd/ReadVariableOp.^sequential_47/dense_140/MatMul/ReadVariableOp/^sequential_47/dense_141/BiasAdd/ReadVariableOp.^sequential_47/dense_141/MatMul/ReadVariableOp4^sequential_47/lstm_116/lstm_cell_116/ReadVariableOp6^sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_16^sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_26^sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_3:^sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOp<^sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOp^sequential_47/lstm_116/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2`
.sequential_47/dense_140/BiasAdd/ReadVariableOp.sequential_47/dense_140/BiasAdd/ReadVariableOp2^
-sequential_47/dense_140/MatMul/ReadVariableOp-sequential_47/dense_140/MatMul/ReadVariableOp2`
.sequential_47/dense_141/BiasAdd/ReadVariableOp.sequential_47/dense_141/BiasAdd/ReadVariableOp2^
-sequential_47/dense_141/MatMul/ReadVariableOp-sequential_47/dense_141/MatMul/ReadVariableOp2j
3sequential_47/lstm_116/lstm_cell_116/ReadVariableOp3sequential_47/lstm_116/lstm_cell_116/ReadVariableOp2n
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_15sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_12n
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_25sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_22n
5sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_35sequential_47/lstm_116/lstm_cell_116/ReadVariableOp_32v
9sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOp9sequential_47/lstm_116/lstm_cell_116/split/ReadVariableOp2z
;sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOp;sequential_47/lstm_116/lstm_cell_116/split_1/ReadVariableOp2<
sequential_47/lstm_116/whilesequential_47/lstm_116/while:U Q
+
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_48
Џ
»
while_cond_3702948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3702948___redundant_placeholder05
1while_while_cond_3702948___redundant_placeholder15
1while_while_cond_3702948___redundant_placeholder25
1while_while_cond_3702948___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
й

ь
lstm_116_while_cond_3703874.
*lstm_116_while_lstm_116_while_loop_counter4
0lstm_116_while_lstm_116_while_maximum_iterations
lstm_116_while_placeholder 
lstm_116_while_placeholder_1 
lstm_116_while_placeholder_2 
lstm_116_while_placeholder_30
,lstm_116_while_less_lstm_116_strided_slice_1G
Clstm_116_while_lstm_116_while_cond_3703874___redundant_placeholder0G
Clstm_116_while_lstm_116_while_cond_3703874___redundant_placeholder1G
Clstm_116_while_lstm_116_while_cond_3703874___redundant_placeholder2G
Clstm_116_while_lstm_116_while_cond_3703874___redundant_placeholder3
lstm_116_while_identity
Э
lstm_116/while/LessLesslstm_116_while_placeholder,lstm_116_while_less_lstm_116_strided_slice_1*
T0*
_output_shapes
: 2
lstm_116/while/Lessx
lstm_116/while/IdentityIdentitylstm_116/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_116/while/Identity";
lstm_116_while_identity lstm_116/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
™
Ј
*__inference_lstm_116_layer_call_fn_3704421

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37035202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЛВ
±	
while_body_3704531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_116_split_readvariableop_resource_0:	АD
5while_lstm_cell_116_split_1_readvariableop_resource_0:	А@
-while_lstm_cell_116_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_116_split_readvariableop_resource:	АB
3while_lstm_cell_116_split_1_readvariableop_resource:	А>
+while_lstm_cell_116_readvariableop_resource:	 АИҐ"while/lstm_cell_116/ReadVariableOpҐ$while/lstm_cell_116/ReadVariableOp_1Ґ$while/lstm_cell_116/ReadVariableOp_2Ґ$while/lstm_cell_116/ReadVariableOp_3Ґ(while/lstm_cell_116/split/ReadVariableOpҐ*while/lstm_cell_116/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemН
#while/lstm_cell_116/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_116/ones_like/ShapeП
#while/lstm_cell_116/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2%
#while/lstm_cell_116/ones_like/Const‘
while/lstm_cell_116/ones_likeFill,while/lstm_cell_116/ones_like/Shape:output:0,while/lstm_cell_116/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ones_likeМ
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_116/split/split_dim…
(while/lstm_cell_116/split/ReadVariableOpReadVariableOp3while_lstm_cell_116_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02*
(while/lstm_cell_116/split/ReadVariableOpч
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:00while/lstm_cell_116/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_116/split 
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMulќ
while/lstm_cell_116/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_1ќ
while/lstm_cell_116/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_2ќ
while/lstm_cell_116/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_3Р
%while/lstm_cell_116/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_116/split_1/split_dimЋ
*while/lstm_cell_116/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_116_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02,
*while/lstm_cell_116/split_1/ReadVariableOpп
while/lstm_cell_116/split_1Split.while/lstm_cell_116/split_1/split_dim:output:02while/lstm_cell_116/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_116/split_1√
while/lstm_cell_116/BiasAddBiasAdd$while/lstm_cell_116/MatMul:product:0$while/lstm_cell_116/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd…
while/lstm_cell_116/BiasAdd_1BiasAdd&while/lstm_cell_116/MatMul_1:product:0$while/lstm_cell_116/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_1…
while/lstm_cell_116/BiasAdd_2BiasAdd&while/lstm_cell_116/MatMul_2:product:0$while/lstm_cell_116/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_2…
while/lstm_cell_116/BiasAdd_3BiasAdd&while/lstm_cell_116/MatMul_3:product:0$while/lstm_cell_116/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/BiasAdd_3®
while/lstm_cell_116/mulMulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mulђ
while/lstm_cell_116/mul_1Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_1ђ
while/lstm_cell_116/mul_2Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_2ђ
while/lstm_cell_116/mul_3Mulwhile_placeholder_2&while/lstm_cell_116/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_3Ј
"while/lstm_cell_116/ReadVariableOpReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_116/ReadVariableOp£
'while/lstm_cell_116/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_116/strided_slice/stackІ
)while/lstm_cell_116/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice/stack_1І
)while/lstm_cell_116/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_116/strided_slice/stack_2ф
!while/lstm_cell_116/strided_sliceStridedSlice*while/lstm_cell_116/ReadVariableOp:value:00while/lstm_cell_116/strided_slice/stack:output:02while/lstm_cell_116/strided_slice/stack_1:output:02while/lstm_cell_116/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_116/strided_sliceЅ
while/lstm_cell_116/MatMul_4MatMulwhile/lstm_cell_116/mul:z:0*while/lstm_cell_116/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_4ї
while/lstm_cell_116/addAddV2$while/lstm_cell_116/BiasAdd:output:0&while/lstm_cell_116/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/addФ
while/lstm_cell_116/SigmoidSigmoidwhile/lstm_cell_116/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoidї
$while/lstm_cell_116/ReadVariableOp_1ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_1І
)while/lstm_cell_116/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_116/strided_slice_1/stackЂ
+while/lstm_cell_116/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_116/strided_slice_1/stack_1Ђ
+while/lstm_cell_116/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_1/stack_2А
#while/lstm_cell_116/strided_slice_1StridedSlice,while/lstm_cell_116/ReadVariableOp_1:value:02while/lstm_cell_116/strided_slice_1/stack:output:04while/lstm_cell_116/strided_slice_1/stack_1:output:04while/lstm_cell_116/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_1≈
while/lstm_cell_116/MatMul_5MatMulwhile/lstm_cell_116/mul_1:z:0,while/lstm_cell_116/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_5Ѕ
while/lstm_cell_116/add_1AddV2&while/lstm_cell_116/BiasAdd_1:output:0&while/lstm_cell_116/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_1Ъ
while/lstm_cell_116/Sigmoid_1Sigmoidwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_1І
while/lstm_cell_116/mul_4Mul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_4ї
$while/lstm_cell_116/ReadVariableOp_2ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_2І
)while/lstm_cell_116/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_116/strided_slice_2/stackЂ
+while/lstm_cell_116/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_116/strided_slice_2/stack_1Ђ
+while/lstm_cell_116/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_2/stack_2А
#while/lstm_cell_116/strided_slice_2StridedSlice,while/lstm_cell_116/ReadVariableOp_2:value:02while/lstm_cell_116/strided_slice_2/stack:output:04while/lstm_cell_116/strided_slice_2/stack_1:output:04while/lstm_cell_116/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_2≈
while/lstm_cell_116/MatMul_6MatMulwhile/lstm_cell_116/mul_2:z:0,while/lstm_cell_116/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_6Ѕ
while/lstm_cell_116/add_2AddV2&while/lstm_cell_116/BiasAdd_2:output:0&while/lstm_cell_116/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_2Н
while/lstm_cell_116/ReluReluwhile/lstm_cell_116/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/ReluЄ
while/lstm_cell_116/mul_5Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_5ѓ
while/lstm_cell_116/add_3AddV2while/lstm_cell_116/mul_4:z:0while/lstm_cell_116/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_3ї
$while/lstm_cell_116/ReadVariableOp_3ReadVariableOp-while_lstm_cell_116_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02&
$while/lstm_cell_116/ReadVariableOp_3І
)while/lstm_cell_116/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_116/strided_slice_3/stackЂ
+while/lstm_cell_116/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_116/strided_slice_3/stack_1Ђ
+while/lstm_cell_116/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_116/strided_slice_3/stack_2А
#while/lstm_cell_116/strided_slice_3StridedSlice,while/lstm_cell_116/ReadVariableOp_3:value:02while/lstm_cell_116/strided_slice_3/stack:output:04while/lstm_cell_116/strided_slice_3/stack_1:output:04while/lstm_cell_116/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_116/strided_slice_3≈
while/lstm_cell_116/MatMul_7MatMulwhile/lstm_cell_116/mul_3:z:0,while/lstm_cell_116/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/MatMul_7Ѕ
while/lstm_cell_116/add_4AddV2&while/lstm_cell_116/BiasAdd_3:output:0&while/lstm_cell_116/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/add_4Ъ
while/lstm_cell_116/Sigmoid_2Sigmoidwhile/lstm_cell_116/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Sigmoid_2С
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/Relu_1Љ
while/lstm_cell_116/mul_6Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_116/mul_6б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_6:z:0*
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
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_116/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_116/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ћ

while/NoOpNoOp#^while/lstm_cell_116/ReadVariableOp%^while/lstm_cell_116/ReadVariableOp_1%^while/lstm_cell_116/ReadVariableOp_2%^while/lstm_cell_116/ReadVariableOp_3)^while/lstm_cell_116/split/ReadVariableOp+^while/lstm_cell_116/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_lstm_cell_116_readvariableop_resource-while_lstm_cell_116_readvariableop_resource_0"l
3while_lstm_cell_116_split_1_readvariableop_resource5while_lstm_cell_116_split_1_readvariableop_resource_0"h
1while_lstm_cell_116_split_readvariableop_resource3while_lstm_cell_116_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2H
"while/lstm_cell_116/ReadVariableOp"while/lstm_cell_116/ReadVariableOp2L
$while/lstm_cell_116/ReadVariableOp_1$while/lstm_cell_116/ReadVariableOp_12L
$while/lstm_cell_116/ReadVariableOp_2$while/lstm_cell_116/ReadVariableOp_22L
$while/lstm_cell_116/ReadVariableOp_3$while/lstm_cell_116/ReadVariableOp_32T
(while/lstm_cell_116/split/ReadVariableOp(while/lstm_cell_116/split/ReadVariableOp2X
*while/lstm_cell_116/split_1/ReadVariableOp*while/lstm_cell_116/split_1/ReadVariableOp: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
™
Ј
*__inference_lstm_116_layer_call_fn_3704410

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_116_layer_call_and_return_conditional_losses_37030822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
„R
–
E__inference_lstm_116_layer_call_and_return_conditional_losses_3702262

inputs(
lstm_cell_116_3702174:	А$
lstm_cell_116_3702176:	А(
lstm_cell_116_3702178:	 А
identityИҐ?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpҐ%lstm_cell_116/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
:€€€€€€€€€ 2
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
B :и2
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
:€€€€€€€€€ 2	
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
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
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
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2І
%lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_116_3702174lstm_cell_116_3702176lstm_cell_116_3702178*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_37021732'
%lstm_cell_116/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter»
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_116_3702174lstm_cell_116_3702176lstm_cell_116_3702178*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3702187*
condR
while_cond_3702186*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeў
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_116_3702174*
_output_shapes
:	А*
dtype02A
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOpб
0lstm_116/lstm_cell_116/kernel/Regularizer/SquareSquareGlstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А22
0lstm_116/lstm_cell_116/kernel/Regularizer/Square≥
/lstm_116/lstm_cell_116/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_116/lstm_cell_116/kernel/Regularizer/Constц
-lstm_116/lstm_cell_116/kernel/Regularizer/SumSum4lstm_116/lstm_cell_116/kernel/Regularizer/Square:y:08lstm_116/lstm_cell_116/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/SumІ
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—821
/lstm_116/lstm_cell_116/kernel/Regularizer/mul/xш
-lstm_116/lstm_cell_116/kernel/Regularizer/mulMul8lstm_116/lstm_cell_116/kernel/Regularizer/mul/x:output:06lstm_116/lstm_cell_116/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_116/lstm_cell_116/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityј
NoOpNoOp@^lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp&^lstm_cell_116/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2В
?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp?lstm_116/lstm_cell_116/kernel/Regularizer/Square/ReadVariableOp2N
%lstm_cell_116/StatefulPartitionedCall%lstm_cell_116/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
»
while_cond_3702483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3702483___redundant_placeholder05
1while_while_cond_3702483___redundant_placeholder15
1while_while_cond_3702483___redundant_placeholder25
1while_while_cond_3702483___redundant_placeholder3
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
@: : : : :€€€€€€€€€ :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ё|
Я
#__inference__traced_restore_3706047
file_prefix3
!assignvariableop_dense_140_kernel:  /
!assignvariableop_1_dense_140_bias: 5
#assignvariableop_2_dense_141_kernel: /
!assignvariableop_3_dense_141_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: C
0assignvariableop_9_lstm_116_lstm_cell_116_kernel:	АN
;assignvariableop_10_lstm_116_lstm_cell_116_recurrent_kernel:	 А>
/assignvariableop_11_lstm_116_lstm_cell_116_bias:	А#
assignvariableop_12_total: #
assignvariableop_13_count: =
+assignvariableop_14_adam_dense_140_kernel_m:  7
)assignvariableop_15_adam_dense_140_bias_m: =
+assignvariableop_16_adam_dense_141_kernel_m: 7
)assignvariableop_17_adam_dense_141_bias_m:K
8assignvariableop_18_adam_lstm_116_lstm_cell_116_kernel_m:	АU
Bassignvariableop_19_adam_lstm_116_lstm_cell_116_recurrent_kernel_m:	 АE
6assignvariableop_20_adam_lstm_116_lstm_cell_116_bias_m:	А=
+assignvariableop_21_adam_dense_140_kernel_v:  7
)assignvariableop_22_adam_dense_140_bias_v: =
+assignvariableop_23_adam_dense_141_kernel_v: 7
)assignvariableop_24_adam_dense_141_bias_v:K
8assignvariableop_25_adam_lstm_116_lstm_cell_116_kernel_v:	АU
Bassignvariableop_26_adam_lstm_116_lstm_cell_116_recurrent_kernel_v:	 АE
6assignvariableop_27_adam_lstm_116_lstm_cell_116_bias_v:	А
identity_29ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9÷
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*в
valueЎB’B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesљ
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

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_dense_140_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_140_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_141_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¶
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_141_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ґ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8™
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9µ
AssignVariableOp_9AssignVariableOp0assignvariableop_9_lstm_116_lstm_cell_116_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10√
AssignVariableOp_10AssignVariableOp;assignvariableop_10_lstm_116_lstm_cell_116_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ј
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_116_lstm_cell_116_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14≥
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_dense_140_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15±
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_140_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16≥
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_141_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17±
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_141_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ј
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_lstm_116_lstm_cell_116_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19 
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_lstm_116_lstm_cell_116_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Њ
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_116_lstm_cell_116_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21≥
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_140_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_140_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≥
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_141_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_141_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ј
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_lstm_116_lstm_cell_116_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26 
AssignVariableOp_26AssignVariableOpBassignvariableop_26_adam_lstm_116_lstm_cell_116_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Њ
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_116_lstm_cell_116_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp∆
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28f
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_29Ѓ
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
К
c
G__inference_reshape_70_layer_call_and_return_conditional_losses_3705590

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
strided_slice/stack_2в
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
Reshape/shape/2†
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ј
serving_default£
A
input_485
serving_default_input_48:0€€€€€€€€€B

reshape_704
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ЫГ
и
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
√
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
ї

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
•
trainable_variables
	variables
regularization_losses
 	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
—
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
 

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
б
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
є

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
":   2dense_140/kernel
: 2dense_140/bias
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
≠

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
":  2dense_141/kernel
:2dense_141/bias
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
≠

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
≠

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
0:.	А2lstm_116/lstm_cell_116/kernel
::8	 А2'lstm_116/lstm_cell_116/recurrent_kernel
*:(А2lstm_116/lstm_cell_116/bias
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
≠

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
':%  2Adam/dense_140/kernel/m
!: 2Adam/dense_140/bias/m
':% 2Adam/dense_141/kernel/m
!:2Adam/dense_141/bias/m
5:3	А2$Adam/lstm_116/lstm_cell_116/kernel/m
?:=	 А2.Adam/lstm_116/lstm_cell_116/recurrent_kernel/m
/:-А2"Adam/lstm_116/lstm_cell_116/bias/m
':%  2Adam/dense_140/kernel/v
!: 2Adam/dense_140/bias/v
':% 2Adam/dense_141/kernel/v
!:2Adam/dense_141/bias/v
5:3	А2$Adam/lstm_116/lstm_cell_116/kernel/v
?:=	 А2.Adam/lstm_116/lstm_cell_116/recurrent_kernel/v
/:-А2"Adam/lstm_116/lstm_cell_116/bias/v
ќBЋ
"__inference__wrapped_model_3702049input_48"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
К2З
/__inference_sequential_47_layer_call_fn_3703174
/__inference_sequential_47_layer_call_fn_3703746
/__inference_sequential_47_layer_call_fn_3703765
/__inference_sequential_47_layer_call_fn_3703620ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
ц2у
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704036
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704371
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703654
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703688ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
Л2И
*__inference_lstm_116_layer_call_fn_3704388
*__inference_lstm_116_layer_call_fn_3704399
*__inference_lstm_116_layer_call_fn_3704410
*__inference_lstm_116_layer_call_fn_3704421’
ћ≤»
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
kwonlydefaults™ 
annotations™ *
 
ч2ф
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704664
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704971
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705214
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705521’
ћ≤»
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
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_dense_140_layer_call_fn_3705530Ґ
Щ≤Х
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
annotations™ *
 
р2н
F__inference_dense_140_layer_call_and_return_conditional_losses_3705541Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_dense_141_layer_call_fn_3705556Ґ
Щ≤Х
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
annotations™ *
 
р2н
F__inference_dense_141_layer_call_and_return_conditional_losses_3705572Ґ
Щ≤Х
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
annotations™ *
 
÷2”
,__inference_reshape_70_layer_call_fn_3705577Ґ
Щ≤Х
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
annotations™ *
 
с2о
G__inference_reshape_70_layer_call_and_return_conditional_losses_3705590Ґ
Щ≤Х
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
annotations™ *
 
і2±
__inference_loss_fn_0_3705601П
З≤Г
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
annotations™ *Ґ 
ЌB 
%__inference_signature_wrapper_3703727input_48"Ф
Н≤Й
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
annotations™ *
 
¶2£
/__inference_lstm_cell_116_layer_call_fn_3705624
/__inference_lstm_cell_116_layer_call_fn_3705641Њ
µ≤±
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
kwonlydefaults™ 
annotations™ *
 
№2ў
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705722
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705835Њ
µ≤±
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
kwonlydefaults™ 
annotations™ *
 
і2±
__inference_loss_fn_1_3705846П
З≤Г
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
annotations™ *Ґ £
"__inference__wrapped_model_3702049}&('5Ґ2
+Ґ(
&К#
input_48€€€€€€€€€
™ ";™8
6

reshape_70(К%

reshape_70€€€€€€€€€¶
F__inference_dense_140_layer_call_and_return_conditional_losses_3705541\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ~
+__inference_dense_140_layer_call_fn_3705530O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ ¶
F__inference_dense_141_layer_call_and_return_conditional_losses_3705572\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
+__inference_dense_141_layer_call_fn_3705556O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€<
__inference_loss_fn_0_3705601Ґ

Ґ 
™ "К <
__inference_loss_fn_1_3705846&Ґ

Ґ 
™ "К ∆
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704664}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ∆
E__inference_lstm_116_layer_call_and_return_conditional_losses_3704971}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ґ
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705214m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ґ
E__inference_lstm_116_layer_call_and_return_conditional_losses_3705521m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ Ю
*__inference_lstm_116_layer_call_fn_3704388p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ю
*__inference_lstm_116_layer_call_fn_3704399p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ О
*__inference_lstm_116_layer_call_fn_3704410`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ О
*__inference_lstm_116_layer_call_fn_3704421`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ ћ
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705722э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€ 
"К
states/1€€€€€€€€€ 
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€ 
EЪB
К
0/1/0€€€€€€€€€ 
К
0/1/1€€€€€€€€€ 
Ъ ћ
J__inference_lstm_cell_116_layer_call_and_return_conditional_losses_3705835э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€ 
"К
states/1€€€€€€€€€ 
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€ 
EЪB
К
0/1/0€€€€€€€€€ 
К
0/1/1€€€€€€€€€ 
Ъ °
/__inference_lstm_cell_116_layer_call_fn_3705624н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€ 
"К
states/1€€€€€€€€€ 
p 
™ "cҐ`
К
0€€€€€€€€€ 
AЪ>
К
1/0€€€€€€€€€ 
К
1/1€€€€€€€€€ °
/__inference_lstm_cell_116_layer_call_fn_3705641н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€ 
"К
states/1€€€€€€€€€ 
p
™ "cҐ`
К
0€€€€€€€€€ 
AЪ>
К
1/0€€€€€€€€€ 
К
1/1€€€€€€€€€ І
G__inference_reshape_70_layer_call_and_return_conditional_losses_3705590\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ 
,__inference_reshape_70_layer_call_fn_3705577O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ѕ
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703654s&('=Ґ:
3Ґ0
&К#
input_48€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ѕ
J__inference_sequential_47_layer_call_and_return_conditional_losses_3703688s&('=Ґ:
3Ґ0
&К#
input_48€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ њ
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704036q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ њ
J__inference_sequential_47_layer_call_and_return_conditional_losses_3704371q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Щ
/__inference_sequential_47_layer_call_fn_3703174f&('=Ґ:
3Ґ0
&К#
input_48€€€€€€€€€
p 

 
™ "К€€€€€€€€€Щ
/__inference_sequential_47_layer_call_fn_3703620f&('=Ґ:
3Ґ0
&К#
input_48€€€€€€€€€
p

 
™ "К€€€€€€€€€Ч
/__inference_sequential_47_layer_call_fn_3703746d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ч
/__inference_sequential_47_layer_call_fn_3703765d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€≥
%__inference_signature_wrapper_3703727Й&('AҐ>
Ґ 
7™4
2
input_48&К#
input_48€€€€€€€€€";™8
6

reshape_70(К%

reshape_70€€€€€€€€€