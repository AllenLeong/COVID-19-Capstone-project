Ќг&
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8С≈%
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:  *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
: *
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

: *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
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
lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_namelstm_10/lstm_cell_10/kernel
М
/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/kernel*
_output_shapes
:	А*
dtype0
І
%lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*6
shared_name'%lstm_10/lstm_cell_10/recurrent_kernel
†
9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_10/lstm_cell_10/recurrent_kernel*
_output_shapes
:	 А*
dtype0
Л
lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_10/lstm_cell_10/bias
Д
-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/bias*
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/m
Б
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:  *
dtype0
А
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
: *
dtype0
И
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_13/kernel/m
Б
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
°
"Adam/lstm_10/lstm_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_10/lstm_cell_10/kernel/m
Ъ
6Adam/lstm_10/lstm_cell_10/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_10/lstm_cell_10/kernel/m*
_output_shapes
:	А*
dtype0
µ
,Adam/lstm_10/lstm_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m
Ѓ
@Adam/lstm_10/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Щ
 Adam/lstm_10/lstm_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_10/lstm_cell_10/bias/m
Т
4Adam/lstm_10/lstm_cell_10/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_10/lstm_cell_10/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/v
Б
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:  *
dtype0
А
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
: *
dtype0
И
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_13/kernel/v
Б
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
°
"Adam/lstm_10/lstm_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_10/lstm_cell_10/kernel/v
Ъ
6Adam/lstm_10/lstm_cell_10/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_10/lstm_cell_10/kernel/v*
_output_shapes
:	А*
dtype0
µ
,Adam/lstm_10/lstm_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v
Ѓ
@Adam/lstm_10/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Щ
 Adam/lstm_10/lstm_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_10/lstm_cell_10/bias/v
Т
4Adam/lstm_10/lstm_cell_10/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_10/lstm_cell_10/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Ј,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*т+
valueи+Bе+ Bё+
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
regularization_losses
	variables
		keras_api


signatures
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
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
 
1
&0
'1
(2
3
4
5
6
≠
trainable_variables

)layers
regularization_losses
*metrics
+layer_regularization_losses
	variables
,layer_metrics
-non_trainable_variables
 
О
.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
 

&0
'1
(2
 

&0
'1
(2
є

3states
trainable_variables

4layers
regularization_losses
5metrics
6layer_regularization_losses
	variables
7layer_metrics
8non_trainable_variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠

9layers
trainable_variables
regularization_losses
:metrics
;layer_regularization_losses
	variables
<layer_metrics
=non_trainable_variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠

>layers
trainable_variables
regularization_losses
?metrics
@layer_regularization_losses
	variables
Alayer_metrics
Bnon_trainable_variables
 
 
 
≠

Clayers
trainable_variables
regularization_losses
Dmetrics
Elayer_regularization_losses
	variables
Flayer_metrics
Gnon_trainable_variables
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
VARIABLE_VALUElstm_10/lstm_cell_10/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_10/lstm_cell_10/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_10/lstm_cell_10/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

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
 

&0
'1
(2
≠

Ilayers
/trainable_variables
0regularization_losses
Jmetrics
Klayer_regularization_losses
1	variables
Llayer_metrics
Mnon_trainable_variables
 
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
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_10/lstm_cell_10/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_10/lstm_cell_10/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_10/lstm_cell_10/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_10/lstm_cell_10/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_10/lstm_cell_10/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_10/lstm_cell_10/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
В
serving_default_input_5Placeholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5lstm_10/lstm_cell_10/kernellstm_10/lstm_cell_10/bias%lstm_10/lstm_cell_10/recurrent_kerneldense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_437319
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOp9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp-lstm_10/lstm_cell_10/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp6Adam/lstm_10/lstm_cell_10/kernel/m/Read/ReadVariableOp@Adam/lstm_10/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_10/lstm_cell_10/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp6Adam/lstm_10/lstm_cell_10/kernel/v/Read/ReadVariableOp@Adam/lstm_10/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_10/lstm_cell_10/bias/v/Read/ReadVariableOpConst*)
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
GPU 2J 8В *(
f#R!
__inference__traced_save_439545
ƒ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biastotalcountAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m"Adam/lstm_10/lstm_cell_10/kernel/m,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m Adam/lstm_10/lstm_cell_10/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v"Adam/lstm_10/lstm_cell_10/kernel/v,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v Adam/lstm_10/lstm_cell_10/bias/v*(
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_439639У∆$
пЌ
Љ
lstm_10_while_body_437770,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	АK
<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	АG
4lstm_10_while_lstm_cell_10_readvariableop_resource_0:	 А
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorK
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:	АI
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	АE
2lstm_10_while_lstm_cell_10_readvariableop_resource:	 АИҐ)lstm_10/while/lstm_cell_10/ReadVariableOpҐ+lstm_10/while/lstm_cell_10/ReadVariableOp_1Ґ+lstm_10/while/lstm_cell_10/ReadVariableOp_2Ґ+lstm_10/while/lstm_cell_10/ReadVariableOp_3Ґ/lstm_10/while/lstm_cell_10/split/ReadVariableOpҐ1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp”
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItem£
*lstm_10/while/lstm_cell_10/ones_like/ShapeShapelstm_10_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_10/while/lstm_cell_10/ones_like/ShapeЭ
*lstm_10/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*lstm_10/while/lstm_cell_10/ones_like/Constр
$lstm_10/while/lstm_cell_10/ones_likeFill3lstm_10/while/lstm_cell_10/ones_like/Shape:output:03lstm_10/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/ones_likeЩ
(lstm_10/while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2*
(lstm_10/while/lstm_cell_10/dropout/Constл
&lstm_10/while/lstm_cell_10/dropout/MulMul-lstm_10/while/lstm_cell_10/ones_like:output:01lstm_10/while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_10/while/lstm_cell_10/dropout/Mul±
(lstm_10/while/lstm_cell_10/dropout/ShapeShape-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_10/while/lstm_cell_10/dropout/Shape°
?lstm_10/while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform1lstm_10/while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2цє%2A
?lstm_10/while/lstm_cell_10/dropout/random_uniform/RandomUniformЂ
1lstm_10/while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>23
1lstm_10/while/lstm_cell_10/dropout/GreaterEqual/y™
/lstm_10/while/lstm_cell_10/dropout/GreaterEqualGreaterEqualHlstm_10/while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:0:lstm_10/while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/lstm_10/while/lstm_cell_10/dropout/GreaterEqual–
'lstm_10/while/lstm_cell_10/dropout/CastCast3lstm_10/while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2)
'lstm_10/while/lstm_cell_10/dropout/Castж
(lstm_10/while/lstm_cell_10/dropout/Mul_1Mul*lstm_10/while/lstm_cell_10/dropout/Mul:z:0+lstm_10/while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_10/while/lstm_cell_10/dropout/Mul_1Э
*lstm_10/while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2,
*lstm_10/while/lstm_cell_10/dropout_1/Constс
(lstm_10/while/lstm_cell_10/dropout_1/MulMul-lstm_10/while/lstm_cell_10/ones_like:output:03lstm_10/while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_10/while/lstm_cell_10/dropout_1/Mulµ
*lstm_10/while/lstm_cell_10/dropout_1/ShapeShape-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_10/while/lstm_cell_10/dropout_1/Shape®
Alstm_10/while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_10/while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Їь«2C
Alstm_10/while/lstm_cell_10/dropout_1/random_uniform/RandomUniformѓ
3lstm_10/while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_10/while/lstm_cell_10/dropout_1/GreaterEqual/y≤
1lstm_10/while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualJlstm_10/while/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0<lstm_10/while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1lstm_10/while/lstm_cell_10/dropout_1/GreaterEqual÷
)lstm_10/while/lstm_cell_10/dropout_1/CastCast5lstm_10/while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_10/while/lstm_cell_10/dropout_1/Castо
*lstm_10/while/lstm_cell_10/dropout_1/Mul_1Mul,lstm_10/while/lstm_cell_10/dropout_1/Mul:z:0-lstm_10/while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_10/while/lstm_cell_10/dropout_1/Mul_1Э
*lstm_10/while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2,
*lstm_10/while/lstm_cell_10/dropout_2/Constс
(lstm_10/while/lstm_cell_10/dropout_2/MulMul-lstm_10/while/lstm_cell_10/ones_like:output:03lstm_10/while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_10/while/lstm_cell_10/dropout_2/Mulµ
*lstm_10/while/lstm_cell_10/dropout_2/ShapeShape-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_10/while/lstm_cell_10/dropout_2/Shape®
Alstm_10/while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_10/while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2™фЙ2C
Alstm_10/while/lstm_cell_10/dropout_2/random_uniform/RandomUniformѓ
3lstm_10/while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_10/while/lstm_cell_10/dropout_2/GreaterEqual/y≤
1lstm_10/while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualJlstm_10/while/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0<lstm_10/while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1lstm_10/while/lstm_cell_10/dropout_2/GreaterEqual÷
)lstm_10/while/lstm_cell_10/dropout_2/CastCast5lstm_10/while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_10/while/lstm_cell_10/dropout_2/Castо
*lstm_10/while/lstm_cell_10/dropout_2/Mul_1Mul,lstm_10/while/lstm_cell_10/dropout_2/Mul:z:0-lstm_10/while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_10/while/lstm_cell_10/dropout_2/Mul_1Э
*lstm_10/while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2,
*lstm_10/while/lstm_cell_10/dropout_3/Constс
(lstm_10/while/lstm_cell_10/dropout_3/MulMul-lstm_10/while/lstm_cell_10/ones_like:output:03lstm_10/while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_10/while/lstm_cell_10/dropout_3/Mulµ
*lstm_10/while/lstm_cell_10/dropout_3/ShapeShape-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_10/while/lstm_cell_10/dropout_3/Shape®
Alstm_10/while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_10/while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2¬у„2C
Alstm_10/while/lstm_cell_10/dropout_3/random_uniform/RandomUniformѓ
3lstm_10/while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>25
3lstm_10/while/lstm_cell_10/dropout_3/GreaterEqual/y≤
1lstm_10/while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualJlstm_10/while/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0<lstm_10/while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1lstm_10/while/lstm_cell_10/dropout_3/GreaterEqual÷
)lstm_10/while/lstm_cell_10/dropout_3/CastCast5lstm_10/while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_10/while/lstm_cell_10/dropout_3/Castо
*lstm_10/while/lstm_cell_10/dropout_3/Mul_1Mul,lstm_10/while/lstm_cell_10/dropout_3/Mul:z:0-lstm_10/while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*lstm_10/while/lstm_cell_10/dropout_3/Mul_1Ъ
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dimё
/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOp:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/lstm_10/while/lstm_cell_10/split/ReadVariableOpУ
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:07lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_10/while/lstm_cell_10/splitз
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_10/while/lstm_cell_10/MatMulл
#lstm_10/while/lstm_cell_10/MatMul_1MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_1л
#lstm_10/while/lstm_cell_10/MatMul_2MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_2л
#lstm_10/while/lstm_cell_10/MatMul_3MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_3Ю
,lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_10/while/lstm_cell_10/split_1/split_dimа
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpЛ
"lstm_10/while/lstm_cell_10/split_1Split5lstm_10/while/lstm_cell_10/split_1/split_dim:output:09lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_10/while/lstm_cell_10/split_1я
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd+lstm_10/while/lstm_cell_10/MatMul:product:0+lstm_10/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/while/lstm_cell_10/BiasAddе
$lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd-lstm_10/while/lstm_cell_10/MatMul_1:product:0+lstm_10/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_1е
$lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd-lstm_10/while/lstm_cell_10/MatMul_2:product:0+lstm_10/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_2е
$lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd-lstm_10/while/lstm_cell_10/MatMul_3:product:0+lstm_10/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_3ƒ
lstm_10/while/lstm_cell_10/mulMullstm_10_while_placeholder_2,lstm_10/while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/while/lstm_cell_10/mul 
 lstm_10/while/lstm_cell_10/mul_1Mullstm_10_while_placeholder_2.lstm_10/while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_1 
 lstm_10/while/lstm_cell_10/mul_2Mullstm_10_while_placeholder_2.lstm_10/while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_2 
 lstm_10/while/lstm_cell_10/mul_3Mullstm_10_while_placeholder_2.lstm_10/while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_3ћ
)lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_10/while/lstm_cell_10/ReadVariableOp±
.lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_10/while/lstm_cell_10/strided_slice/stackµ
0lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_10/while/lstm_cell_10/strided_slice/stack_1µ
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Ю
(lstm_10/while/lstm_cell_10/strided_sliceStridedSlice1lstm_10/while/lstm_cell_10/ReadVariableOp:value:07lstm_10/while/lstm_cell_10/strided_slice/stack:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_10/while/lstm_cell_10/strided_sliceЁ
#lstm_10/while/lstm_cell_10/MatMul_4MatMul"lstm_10/while/lstm_cell_10/mul:z:01lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_4„
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/BiasAdd:output:0-lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/while/lstm_cell_10/add©
"lstm_10/while/lstm_cell_10/SigmoidSigmoid"lstm_10/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/while/lstm_cell_10/Sigmoid–
+lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_1µ
0lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_10/while/lstm_cell_10/strided_slice_1/stackє
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_1StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:09lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_1б
#lstm_10/while/lstm_cell_10/MatMul_5MatMul$lstm_10/while/lstm_cell_10/mul_1:z:03lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_5Ё
 lstm_10/while/lstm_cell_10/add_1AddV2-lstm_10/while/lstm_cell_10/BiasAdd_1:output:0-lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_1ѓ
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/Sigmoid_1ƒ
 lstm_10/while/lstm_cell_10/mul_4Mul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_4–
+lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_2µ
0lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_10/while/lstm_cell_10/strided_slice_2/stackє
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_2StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:09lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_2б
#lstm_10/while/lstm_cell_10/MatMul_6MatMul$lstm_10/while/lstm_cell_10/mul_2:z:03lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_6Ё
 lstm_10/while/lstm_cell_10/add_2AddV2-lstm_10/while/lstm_cell_10/BiasAdd_2:output:0-lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_2Ґ
lstm_10/while/lstm_cell_10/ReluRelu$lstm_10/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_10/while/lstm_cell_10/Relu‘
 lstm_10/while/lstm_cell_10/mul_5Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_5Ћ
 lstm_10/while/lstm_cell_10/add_3AddV2$lstm_10/while/lstm_cell_10/mul_4:z:0$lstm_10/while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_3–
+lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_3µ
0lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_10/while/lstm_cell_10/strided_slice_3/stackє
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_3StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:09lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_3б
#lstm_10/while/lstm_cell_10/MatMul_7MatMul$lstm_10/while/lstm_cell_10/mul_3:z:03lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_7Ё
 lstm_10/while/lstm_cell_10/add_4AddV2-lstm_10/while/lstm_cell_10/BiasAdd_3:output:0-lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_4ѓ
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid$lstm_10/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/Sigmoid_2¶
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_10/while/lstm_cell_10/Relu_1Ў
 lstm_10/while/lstm_cell_10/mul_6Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_6И
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_10/while/TensorArrayV2Write/TensorListSetIteml
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add/yЙ
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/addp
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add_1/yЮ
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1Л
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity¶
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1Н
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2Ї
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3≠
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_6:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/while/Identity_4≠
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_3:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/while/Identity_5Ж
lstm_10/while/NoOpNoOp*^lstm_10/while/lstm_cell_10/ReadVariableOp,^lstm_10/while/lstm_cell_10/ReadVariableOp_1,^lstm_10/while/lstm_cell_10/ReadVariableOp_2,^lstm_10/while/lstm_cell_10/ReadVariableOp_30^lstm_10/while/lstm_cell_10/split/ReadVariableOp2^lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_10/while/NoOp"9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"j
2lstm_10_while_lstm_cell_10_readvariableop_resource4lstm_10_while_lstm_cell_10_readvariableop_resource_0"z
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"v
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"»
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)lstm_10/while/lstm_cell_10/ReadVariableOp)lstm_10/while/lstm_cell_10/ReadVariableOp2Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_1+lstm_10/while/lstm_cell_10/ReadVariableOp_12Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_2+lstm_10/while/lstm_cell_10/ReadVariableOp_22Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_3+lstm_10/while/lstm_cell_10/ReadVariableOp_32b
/lstm_10/while/lstm_cell_10/split/ReadVariableOp/lstm_10/while/lstm_cell_10/split/ReadVariableOp2f
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp: 
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
’
√
while_cond_436075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_436075___redundant_placeholder04
0while_while_cond_436075___redundant_placeholder14
0while_while_cond_436075___redundant_placeholder24
0while_while_cond_436075___redundant_placeholder3
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
с
Ц
)__inference_dense_12_layer_call_fn_439122

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallф
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
GPU 2J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4366932
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
жа
Ъ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437628

inputsE
2lstm_10_lstm_cell_10_split_readvariableop_resource:	АC
4lstm_10_lstm_cell_10_split_1_readvariableop_resource:	А?
,lstm_10_lstm_cell_10_readvariableop_resource:	 А9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource: 6
(dense_13_biasadd_readvariableop_resource:
identityИҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐ#lstm_10/lstm_cell_10/ReadVariableOpҐ%lstm_10/lstm_cell_10/ReadVariableOp_1Ґ%lstm_10/lstm_cell_10/ReadVariableOp_2Ґ%lstm_10/lstm_cell_10/ReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ)lstm_10/lstm_cell_10/split/ReadVariableOpҐ+lstm_10/lstm_cell_10/split_1/ReadVariableOpҐlstm_10/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/ShapeД
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stackИ
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1И
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2Т
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicel
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros/mul/yМ
lstm_10/zeros/mulMullstm_10/strided_slice:output:0lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/mulo
lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_10/zeros/Less/yЗ
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lessr
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros/packed/1£
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros/packedo
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros/ConstХ
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/zerosp
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros_1/mul/yТ
lstm_10/zeros_1/mulMullstm_10/strided_slice:output:0lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/muls
lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_10/zeros_1/Less/yП
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessv
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros_1/packed/1©
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros_1/packeds
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros_1/ConstЭ
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/zeros_1Е
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/permТ
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1И
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stackМ
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1М
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2Ю
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1Х
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#lstm_10/TensorArrayV2/element_shape“
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2ѕ
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensorИ
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stackМ
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1М
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2ђ
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_10/strided_slice_2Т
$lstm_10/lstm_cell_10/ones_like/ShapeShapelstm_10/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_10/lstm_cell_10/ones_like/ShapeС
$lstm_10/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm_10/lstm_cell_10/ones_like/ConstЎ
lstm_10/lstm_cell_10/ones_likeFill-lstm_10/lstm_cell_10/ones_like/Shape:output:0-lstm_10/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/ones_likeО
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dim 
)lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)lstm_10/lstm_cell_10/split/ReadVariableOpы
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:01lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_10/lstm_cell_10/splitљ
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMulЅ
lstm_10/lstm_cell_10/MatMul_1MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_1Ѕ
lstm_10/lstm_cell_10/MatMul_2MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_2Ѕ
lstm_10/lstm_cell_10/MatMul_3MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_3Т
&lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_10/lstm_cell_10/split_1/split_dimћ
+lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_10/lstm_cell_10/split_1/ReadVariableOpу
lstm_10/lstm_cell_10/split_1Split/lstm_10/lstm_cell_10/split_1/split_dim:output:03lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_10/lstm_cell_10/split_1«
lstm_10/lstm_cell_10/BiasAddBiasAdd%lstm_10/lstm_cell_10/MatMul:product:0%lstm_10/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/BiasAddЌ
lstm_10/lstm_cell_10/BiasAdd_1BiasAdd'lstm_10/lstm_cell_10/MatMul_1:product:0%lstm_10/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_1Ќ
lstm_10/lstm_cell_10/BiasAdd_2BiasAdd'lstm_10/lstm_cell_10/MatMul_2:product:0%lstm_10/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_2Ќ
lstm_10/lstm_cell_10/BiasAdd_3BiasAdd'lstm_10/lstm_cell_10/MatMul_3:product:0%lstm_10/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_3Ѓ
lstm_10/lstm_cell_10/mulMullstm_10/zeros:output:0'lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul≤
lstm_10/lstm_cell_10/mul_1Mullstm_10/zeros:output:0'lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_1≤
lstm_10/lstm_cell_10/mul_2Mullstm_10/zeros:output:0'lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_2≤
lstm_10/lstm_cell_10/mul_3Mullstm_10/zeros:output:0'lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_3Є
#lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_10/lstm_cell_10/ReadVariableOp•
(lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_10/lstm_cell_10/strided_slice/stack©
*lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_10/lstm_cell_10/strided_slice/stack_1©
*lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_10/lstm_cell_10/strided_slice/stack_2ъ
"lstm_10/lstm_cell_10/strided_sliceStridedSlice+lstm_10/lstm_cell_10/ReadVariableOp:value:01lstm_10/lstm_cell_10/strided_slice/stack:output:03lstm_10/lstm_cell_10/strided_slice/stack_1:output:03lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_10/lstm_cell_10/strided_slice≈
lstm_10/lstm_cell_10/MatMul_4MatMullstm_10/lstm_cell_10/mul:z:0+lstm_10/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_4њ
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/BiasAdd:output:0'lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/addЧ
lstm_10/lstm_cell_10/SigmoidSigmoidlstm_10/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/SigmoidЉ
%lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_1©
*lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_10/lstm_cell_10/strided_slice_1/stack≠
,lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_10/lstm_cell_10/strided_slice_1/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_1StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_1:value:03lstm_10/lstm_cell_10/strided_slice_1/stack:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_1…
lstm_10/lstm_cell_10/MatMul_5MatMullstm_10/lstm_cell_10/mul_1:z:0-lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_5≈
lstm_10/lstm_cell_10/add_1AddV2'lstm_10/lstm_cell_10/BiasAdd_1:output:0'lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_1Э
lstm_10/lstm_cell_10/Sigmoid_1Sigmoidlstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/Sigmoid_1ѓ
lstm_10/lstm_cell_10/mul_4Mul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_4Љ
%lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_2©
*lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_10/lstm_cell_10/strided_slice_2/stack≠
,lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_10/lstm_cell_10/strided_slice_2/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_2StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_2:value:03lstm_10/lstm_cell_10/strided_slice_2/stack:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_2…
lstm_10/lstm_cell_10/MatMul_6MatMullstm_10/lstm_cell_10/mul_2:z:0-lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_6≈
lstm_10/lstm_cell_10/add_2AddV2'lstm_10/lstm_cell_10/BiasAdd_2:output:0'lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_2Р
lstm_10/lstm_cell_10/ReluRelulstm_10/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/ReluЉ
lstm_10/lstm_cell_10/mul_5Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_5≥
lstm_10/lstm_cell_10/add_3AddV2lstm_10/lstm_cell_10/mul_4:z:0lstm_10/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_3Љ
%lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_3©
*lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_10/lstm_cell_10/strided_slice_3/stack≠
,lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_10/lstm_cell_10/strided_slice_3/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_3StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_3:value:03lstm_10/lstm_cell_10/strided_slice_3/stack:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_3…
lstm_10/lstm_cell_10/MatMul_7MatMullstm_10/lstm_cell_10/mul_3:z:0-lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_7≈
lstm_10/lstm_cell_10/add_4AddV2'lstm_10/lstm_cell_10/BiasAdd_3:output:0'lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_4Э
lstm_10/lstm_cell_10/Sigmoid_2Sigmoidlstm_10/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/Sigmoid_2Ф
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/Relu_1ј
lstm_10/lstm_cell_10/mul_6Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_6Я
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2'
%lstm_10/TensorArrayV2_1/element_shapeЎ
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2_1^
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/timeП
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counterщ
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_10_lstm_cell_10_split_readvariableop_resource4lstm_10_lstm_cell_10_split_1_readvariableop_resource,lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_10_while_body_437467*%
condR
lstm_10_while_cond_437466*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_10/while≈
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStackС
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_10/strided_slice_3/stackМ
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1М
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2 
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_10/strided_slice_3Й
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/perm≈
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_10/transpose_1v
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/runtime®
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_12/MatMul/ReadVariableOp®
dense_12/MatMulMatMul lstm_10/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/Relu®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/BiasAddk
reshape_6/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_6/ShapeИ
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stackМ
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1М
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2Ю
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_6/strided_slicex
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/1x
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/2“
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shape§
reshape_6/ReshapeReshapedense_13/BiasAdd:output:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_6/Reshapeт
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mul«
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/muly
IdentityIdentityreshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityќ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp0^dense_13/bias/Regularizer/Square/ReadVariableOp$^lstm_10/lstm_cell_10/ReadVariableOp&^lstm_10/lstm_cell_10/ReadVariableOp_1&^lstm_10/lstm_cell_10/ReadVariableOp_2&^lstm_10/lstm_cell_10/ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*^lstm_10/lstm_cell_10/split/ReadVariableOp,^lstm_10/lstm_cell_10/split_1/ReadVariableOp^lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2J
#lstm_10/lstm_cell_10/ReadVariableOp#lstm_10/lstm_cell_10/ReadVariableOp2N
%lstm_10/lstm_cell_10/ReadVariableOp_1%lstm_10/lstm_cell_10/ReadVariableOp_12N
%lstm_10/lstm_cell_10/ReadVariableOp_2%lstm_10/lstm_cell_10/ReadVariableOp_22N
%lstm_10/lstm_cell_10/ReadVariableOp_3%lstm_10/lstm_cell_10/ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_10/lstm_cell_10/split/ReadVariableOp)lstm_10/lstm_cell_10/split/ReadVariableOp2Z
+lstm_10/lstm_cell_10/split_1/ReadVariableOp+lstm_10/lstm_cell_10/split_1/ReadVariableOp2
lstm_10/whilelstm_10/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ў+
™
H__inference_sequential_4_layer_call_and_return_conditional_losses_436749

inputs!
lstm_10_436675:	А
lstm_10_436677:	А!
lstm_10_436679:	 А!
dense_12_436694:  
dense_12_436696: !
dense_13_436716: 
dense_13_436718:
identityИҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐlstm_10/StatefulPartitionedCallҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp°
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_436675lstm_10_436677lstm_10_436679*
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4366742!
lstm_10/StatefulPartitionedCallґ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_12_436694dense_12_436696*
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
GPU 2J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4366932"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_436716dense_13_436718*
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
GPU 2J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4367152"
 dense_13/StatefulPartitionedCallю
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_4367342
reshape_6/PartitionedCallќ
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_10_436675*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mulЃ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_436718*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulБ
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identity®
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall0^dense_13/bias/Regularizer/Square/ReadVariableOp ^lstm_10/StatefulPartitionedCall>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_435778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_435778___redundant_placeholder04
0while_while_cond_435778___redundant_placeholder14
0while_while_cond_435778___redundant_placeholder24
0while_while_cond_435778___redundant_placeholder3
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
∆ґ
Д
&sequential_4_lstm_10_while_body_435492F
Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counterL
Hsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations*
&sequential_4_lstm_10_while_placeholder,
(sequential_4_lstm_10_while_placeholder_1,
(sequential_4_lstm_10_while_placeholder_2,
(sequential_4_lstm_10_while_placeholder_3E
Asequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1_0Б
}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_4_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	АX
Isequential_4_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	АT
Asequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0:	 А'
#sequential_4_lstm_10_while_identity)
%sequential_4_lstm_10_while_identity_1)
%sequential_4_lstm_10_while_identity_2)
%sequential_4_lstm_10_while_identity_3)
%sequential_4_lstm_10_while_identity_4)
%sequential_4_lstm_10_while_identity_5C
?sequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1
{sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensorX
Esequential_4_lstm_10_while_lstm_cell_10_split_readvariableop_resource:	АV
Gsequential_4_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	АR
?sequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource:	 АИҐ6sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOpҐ8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_1Ґ8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_2Ґ8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_3Ґ<sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOpҐ>sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOpн
Lsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2N
Lsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape—
>sequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_10_while_placeholderUsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02@
>sequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem 
7sequential_4/lstm_10/while/lstm_cell_10/ones_like/ShapeShape(sequential_4_lstm_10_while_placeholder_2*
T0*
_output_shapes
:29
7sequential_4/lstm_10/while/lstm_cell_10/ones_like/ShapeЈ
7sequential_4/lstm_10/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7sequential_4/lstm_10/while/lstm_cell_10/ones_like/Const§
1sequential_4/lstm_10/while/lstm_cell_10/ones_likeFill@sequential_4/lstm_10/while/lstm_cell_10/ones_like/Shape:output:0@sequential_4/lstm_10/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/ones_likeі
7sequential_4/lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_4/lstm_10/while/lstm_cell_10/split/split_dimЕ
<sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOpGsequential_4_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02>
<sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOp«
-sequential_4/lstm_10/while/lstm_cell_10/splitSplit@sequential_4/lstm_10/while/lstm_cell_10/split/split_dim:output:0Dsequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2/
-sequential_4/lstm_10/while/lstm_cell_10/splitЫ
.sequential_4/lstm_10/while/lstm_cell_10/MatMulMatMulEsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_4/lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_4/lstm_10/while/lstm_cell_10/MatMulЯ
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_1MatMulEsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_4/lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_1Я
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_2MatMulEsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_4/lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_2Я
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_3MatMulEsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_4/lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_3Є
9sequential_4/lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9sequential_4/lstm_10/while/lstm_cell_10/split_1/split_dimЗ
>sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOpIsequential_4_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02@
>sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOpњ
/sequential_4/lstm_10/while/lstm_cell_10/split_1SplitBsequential_4/lstm_10/while/lstm_cell_10/split_1/split_dim:output:0Fsequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split21
/sequential_4/lstm_10/while/lstm_cell_10/split_1У
/sequential_4/lstm_10/while/lstm_cell_10/BiasAddBiasAdd8sequential_4/lstm_10/while/lstm_cell_10/MatMul:product:08sequential_4/lstm_10/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_4/lstm_10/while/lstm_cell_10/BiasAddЩ
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd:sequential_4/lstm_10/while/lstm_cell_10/MatMul_1:product:08sequential_4/lstm_10/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_1Щ
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd:sequential_4/lstm_10/while/lstm_cell_10/MatMul_2:product:08sequential_4/lstm_10/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_2Щ
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd:sequential_4/lstm_10/while/lstm_cell_10/MatMul_3:product:08sequential_4/lstm_10/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_3щ
+sequential_4/lstm_10/while/lstm_cell_10/mulMul(sequential_4_lstm_10_while_placeholder_2:sequential_4/lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/while/lstm_cell_10/mulэ
-sequential_4/lstm_10/while/lstm_cell_10/mul_1Mul(sequential_4_lstm_10_while_placeholder_2:sequential_4/lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_1э
-sequential_4/lstm_10/while/lstm_cell_10/mul_2Mul(sequential_4_lstm_10_while_placeholder_2:sequential_4/lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_2э
-sequential_4/lstm_10/while/lstm_cell_10/mul_3Mul(sequential_4_lstm_10_while_placeholder_2:sequential_4/lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_3у
6sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOpAsequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype028
6sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOpЋ
;sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stackѕ
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_1ѕ
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_2м
5sequential_4/lstm_10/while/lstm_cell_10/strided_sliceStridedSlice>sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp:value:0Dsequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack:output:0Fsequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:0Fsequential_4/lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_4/lstm_10/while/lstm_cell_10/strided_sliceС
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_4MatMul/sequential_4/lstm_10/while/lstm_cell_10/mul:z:0>sequential_4/lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_4Л
+sequential_4/lstm_10/while/lstm_cell_10/addAddV28sequential_4/lstm_10/while/lstm_cell_10/BiasAdd:output:0:sequential_4/lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/while/lstm_cell_10/add–
/sequential_4/lstm_10/while/lstm_cell_10/SigmoidSigmoid/sequential_4/lstm_10/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_4/lstm_10/while/lstm_cell_10/Sigmoidч
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOpAsequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02:
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_1ѕ
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_1”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_2ш
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1StridedSlice@sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:0Fsequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1Х
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_5MatMul1sequential_4/lstm_10/while/lstm_cell_10/mul_1:z:0@sequential_4/lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_5С
-sequential_4/lstm_10/while/lstm_cell_10/add_1AddV2:sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_1:output:0:sequential_4/lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/add_1÷
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid1sequential_4/lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_1ш
-sequential_4/lstm_10/while/lstm_cell_10/mul_4Mul5sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_1:y:0(sequential_4_lstm_10_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_4ч
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOpAsequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02:
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_2ѕ
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2?
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_1”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_2ш
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2StridedSlice@sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:0Fsequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2Х
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_6MatMul1sequential_4/lstm_10/while/lstm_cell_10/mul_2:z:0@sequential_4/lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_6С
-sequential_4/lstm_10/while/lstm_cell_10/add_2AddV2:sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_2:output:0:sequential_4/lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/add_2…
,sequential_4/lstm_10/while/lstm_cell_10/ReluRelu1sequential_4/lstm_10/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_4/lstm_10/while/lstm_cell_10/ReluИ
-sequential_4/lstm_10/while/lstm_cell_10/mul_5Mul3sequential_4/lstm_10/while/lstm_cell_10/Sigmoid:y:0:sequential_4/lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_5€
-sequential_4/lstm_10/while/lstm_cell_10/add_3AddV21sequential_4/lstm_10/while/lstm_cell_10/mul_4:z:01sequential_4/lstm_10/while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/add_3ч
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOpAsequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02:
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_3ѕ
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2?
=sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_1”
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_2ш
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3StridedSlice@sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:0Fsequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0Hsequential_4/lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3Х
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_7MatMul1sequential_4/lstm_10/while/lstm_cell_10/mul_3:z:0@sequential_4/lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 22
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_7С
-sequential_4/lstm_10/while/lstm_cell_10/add_4AddV2:sequential_4/lstm_10/while/lstm_cell_10/BiasAdd_3:output:0:sequential_4/lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/add_4÷
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid1sequential_4/lstm_10/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 23
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_2Ќ
.sequential_4/lstm_10/while/lstm_cell_10/Relu_1Relu1sequential_4/lstm_10/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_4/lstm_10/while/lstm_cell_10/Relu_1М
-sequential_4/lstm_10/while/lstm_cell_10/mul_6Mul5sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_2:y:0<sequential_4/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_4/lstm_10/while/lstm_cell_10/mul_6…
?sequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_10_while_placeholder_1&sequential_4_lstm_10_while_placeholder1sequential_4/lstm_10/while/lstm_cell_10/mul_6:z:0*
_output_shapes
: *
element_dtype02A
?sequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_4/lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_4/lstm_10/while/add/yљ
sequential_4/lstm_10/while/addAddV2&sequential_4_lstm_10_while_placeholder)sequential_4/lstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_10/while/addК
"sequential_4/lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_4/lstm_10/while/add_1/yя
 sequential_4/lstm_10/while/add_1AddV2Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counter+sequential_4/lstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_10/while/add_1њ
#sequential_4/lstm_10/while/IdentityIdentity$sequential_4/lstm_10/while/add_1:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_4/lstm_10/while/Identityз
%sequential_4/lstm_10/while/Identity_1IdentityHsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_10/while/Identity_1Ѕ
%sequential_4/lstm_10/while/Identity_2Identity"sequential_4/lstm_10/while/add:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_10/while/Identity_2о
%sequential_4/lstm_10/while/Identity_3IdentityOsequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_10/while/Identity_3б
%sequential_4/lstm_10/while/Identity_4Identity1sequential_4/lstm_10/while/lstm_cell_10/mul_6:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_4/lstm_10/while/Identity_4б
%sequential_4/lstm_10/while/Identity_5Identity1sequential_4/lstm_10/while/lstm_cell_10/add_3:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_4/lstm_10/while/Identity_5о
sequential_4/lstm_10/while/NoOpNoOp7^sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp9^sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_19^sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_29^sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_3=^sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOp?^sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_4/lstm_10/while/NoOp"S
#sequential_4_lstm_10_while_identity,sequential_4/lstm_10/while/Identity:output:0"W
%sequential_4_lstm_10_while_identity_1.sequential_4/lstm_10/while/Identity_1:output:0"W
%sequential_4_lstm_10_while_identity_2.sequential_4/lstm_10/while/Identity_2:output:0"W
%sequential_4_lstm_10_while_identity_3.sequential_4/lstm_10/while/Identity_3:output:0"W
%sequential_4_lstm_10_while_identity_4.sequential_4/lstm_10/while/Identity_4:output:0"W
%sequential_4_lstm_10_while_identity_5.sequential_4/lstm_10/while/Identity_5:output:0"Д
?sequential_4_lstm_10_while_lstm_cell_10_readvariableop_resourceAsequential_4_lstm_10_while_lstm_cell_10_readvariableop_resource_0"Ф
Gsequential_4_lstm_10_while_lstm_cell_10_split_1_readvariableop_resourceIsequential_4_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"Р
Esequential_4_lstm_10_while_lstm_cell_10_split_readvariableop_resourceGsequential_4_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"Д
?sequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1Asequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1_0"ь
{sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2p
6sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp6sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp2t
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_18sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_12t
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_28sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_22t
8sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_38sequential_4/lstm_10/while/lstm_cell_10/ReadVariableOp_32|
<sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOp<sequential_4/lstm_10/while/lstm_cell_10/split/ReadVariableOp2А
>sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp>sequential_4/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp: 
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
ќR
й
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_435765

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6Ё
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muld
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

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
И
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_436734

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
’
√
while_cond_438397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_438397___redundant_placeholder04
0while_while_cond_438397___redundant_placeholder14
0while_while_cond_438397___redundant_placeholder24
0while_while_cond_438397___redundant_placeholder3
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
г	
І
-__inference_sequential_4_layer_call_fn_436766
input_5
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_4367492
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
ґv
л
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439427

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
seed2ЧЭг2&
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
seed2£пЖ2(
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
seed2ЭБG2(
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
seed2юЇ≠2(
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6Ё
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muld
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

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
¶
µ
(__inference_lstm_10_layer_call_fn_438002

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallА
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4366742
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_438672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_438672___redundant_placeholder04
0while_while_cond_438672___redundant_placeholder14
0while_while_cond_438672___redundant_placeholder24
0while_while_cond_438672___redundant_placeholder3
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
МR
≈
C__inference_lstm_10_layer_call_and_return_conditional_losses_436151

inputs&
lstm_cell_10_436063:	А"
lstm_cell_10_436065:	А&
lstm_cell_10_436067:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ$lstm_cell_10/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Э
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_436063lstm_cell_10_436065lstm_cell_10_436067*
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4359982&
$lstm_cell_10/StatefulPartitionedCallП
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
while/loop_counterј
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_436063lstm_cell_10_436065lstm_cell_10_436067*
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
bodyR
while_body_436076*
condR
while_cond_436075*K
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
runtime”
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_436063*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityљ
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ФҐ
©
C__inference_lstm_10_layer_call_and_return_conditional_losses_438256
inputs_0=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3О
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulТ
lstm_cell_10/mul_1Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1Т
lstm_cell_10/mul_2Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2Т
lstm_cell_10/mul_3Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_438123*
condR
while_cond_438122*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
џ+
Ђ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437280
input_5!
lstm_10_437249:	А
lstm_10_437251:	А!
lstm_10_437253:	 А!
dense_12_437256:  
dense_12_437258: !
dense_13_437261: 
dense_13_437263:
identityИҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐlstm_10/StatefulPartitionedCallҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_10_437249lstm_10_437251lstm_10_437253*
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4371122!
lstm_10/StatefulPartitionedCallґ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_12_437256dense_12_437258*
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
GPU 2J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4366932"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_437261dense_13_437263*
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
GPU 2J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4367152"
 dense_13/StatefulPartitionedCallю
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_4367342
reshape_6/PartitionedCallќ
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_10_437249*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mulЃ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_437263*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulБ
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identity®
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall0^dense_13/bias/Regularizer/Square/ReadVariableOp ^lstm_10/StatefulPartitionedCall>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
И
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_439182

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
∆
F
*__inference_reshape_6_layer_call_fn_439169

inputs
identity«
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
GPU 2J 8В *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_4367342
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
м
І
D__inference_dense_13_layer_call_and_return_conditional_losses_439164

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ/dense_13/bias/Regularizer/Square/ReadVariableOpН
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
BiasAddЊ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_13/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ћB
в
__inference__traced_save_439545
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableopD
@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop8
4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableopA
=savev2_adam_lstm_10_lstm_cell_10_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_10_lstm_cell_10_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_10_lstm_cell_10_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableopA
=savev2_adam_lstm_10_lstm_cell_10_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_10_lstm_cell_10_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_10_lstm_cell_10_bias_v_read_readvariableop
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
SaveV2/shape_and_slices№
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableop@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop=savev2_adam_lstm_10_lstm_cell_10_kernel_m_read_readvariableopGsavev2_adam_lstm_10_lstm_cell_10_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_10_lstm_cell_10_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop=savev2_adam_lstm_10_lstm_cell_10_kernel_v_read_readvariableopGsavev2_adam_lstm_10_lstm_cell_10_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_10_lstm_cell_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
…: :  : : :: : : : : :	А:	 А:А: : :  : : ::	А:	 А:А:  : : ::	А:	 А:А: 2(
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
:	А:%!

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
:	А:%!

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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:

_output_shapes
: 
ІА
§	
while_body_436541
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3•
while/lstm_cell_10/mulMulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul©
while/lstm_cell_10/mul_1Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1©
while/lstm_cell_10/mul_2Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2©
while/lstm_cell_10/mul_3Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
’
√
while_cond_438122
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_438122___redundant_placeholder04
0while_while_cond_438122___redundant_placeholder14
0while_while_cond_438122___redundant_placeholder24
0while_while_cond_438122___redundant_placeholder3
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
Ўѕ
І
C__inference_lstm_10_layer_call_and_return_conditional_losses_437112

inputs=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout/Const≥
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/MulЗ
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeш
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ІБ§23
1lstm_cell_10/dropout/random_uniform/RandomUniformП
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_10/dropout/GreaterEqual/yт
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_cell_10/dropout/GreaterEqual¶
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/CastЃ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/Mul_1Б
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_1/Constє
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/MulЛ
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shapeю
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Њѕє25
3lstm_cell_10/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_1/GreaterEqual/yъ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_1/GreaterEqualђ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Castґ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Mul_1Б
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_2/Constє
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/MulЛ
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shapeэ
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2у©$25
3lstm_cell_10/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_2/GreaterEqual/yъ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_2/GreaterEqualђ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Castґ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Mul_1Б
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_3/Constє
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/MulЛ
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shapeю
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2∞ре25
3lstm_cell_10/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_3/GreaterEqual/yъ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_3/GreaterEqualђ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Castґ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3Н
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulУ
lstm_cell_10/mul_1Mulzeros:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1У
lstm_cell_10/mul_2Mulzeros:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2У
lstm_cell_10/mul_3Mulzeros:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_436947*
condR
while_cond_436946*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥	
Ю
$__inference_signature_wrapper_437319
input_5
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В **
f%R#
!__inference__wrapped_model_4356412
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
МR
≈
C__inference_lstm_10_layer_call_and_return_conditional_losses_435854

inputs&
lstm_cell_10_435766:	А"
lstm_cell_10_435768:	А&
lstm_cell_10_435770:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ$lstm_cell_10/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Э
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_435766lstm_cell_10_435768lstm_cell_10_435770*
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4357652&
$lstm_cell_10/StatefulPartitionedCallП
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
while/loop_counterј
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_435766lstm_cell_10_435768lstm_cell_10_435770*
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
bodyR
while_body_435779*
condR
while_cond_435778*K
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
runtime”
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_435766*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityљ
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∆

г
lstm_10_while_cond_437769,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1D
@lstm_10_while_lstm_10_while_cond_437769___redundant_placeholder0D
@lstm_10_while_lstm_10_while_cond_437769___redundant_placeholder1D
@lstm_10_while_lstm_10_while_cond_437769___redundant_placeholder2D
@lstm_10_while_lstm_10_while_cond_437769___redundant_placeholder3
lstm_10_while_identity
Ш
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2
lstm_10/while/Lessu
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_10/while/Identity"9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
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
ІА
§	
while_body_438123
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3•
while/lstm_cell_10/mulMulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul©
while/lstm_cell_10/mul_1Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1©
while/lstm_cell_10/mul_2Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2©
while/lstm_cell_10/mul_3Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
В
х
D__inference_dense_12_layer_call_and_return_conditional_losses_436693

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
ВХ
Ъ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437963

inputsE
2lstm_10_lstm_cell_10_split_readvariableop_resource:	АC
4lstm_10_lstm_cell_10_split_1_readvariableop_resource:	А?
,lstm_10_lstm_cell_10_readvariableop_resource:	 А9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource: 6
(dense_13_biasadd_readvariableop_resource:
identityИҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐ#lstm_10/lstm_cell_10/ReadVariableOpҐ%lstm_10/lstm_cell_10/ReadVariableOp_1Ґ%lstm_10/lstm_cell_10/ReadVariableOp_2Ґ%lstm_10/lstm_cell_10/ReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ)lstm_10/lstm_cell_10/split/ReadVariableOpҐ+lstm_10/lstm_cell_10/split_1/ReadVariableOpҐlstm_10/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/ShapeД
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stackИ
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1И
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2Т
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicel
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros/mul/yМ
lstm_10/zeros/mulMullstm_10/strided_slice:output:0lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/mulo
lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_10/zeros/Less/yЗ
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lessr
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros/packed/1£
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros/packedo
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros/ConstХ
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/zerosp
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros_1/mul/yТ
lstm_10/zeros_1/mulMullstm_10/strided_slice:output:0lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/muls
lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_10/zeros_1/Less/yП
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessv
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/zeros_1/packed/1©
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros_1/packeds
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros_1/ConstЭ
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/zeros_1Е
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/permТ
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1И
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stackМ
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1М
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2Ю
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1Х
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#lstm_10/TensorArrayV2/element_shape“
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2ѕ
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensorИ
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stackМ
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1М
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2ђ
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_10/strided_slice_2Т
$lstm_10/lstm_cell_10/ones_like/ShapeShapelstm_10/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_10/lstm_cell_10/ones_like/ShapeС
$lstm_10/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm_10/lstm_cell_10/ones_like/ConstЎ
lstm_10/lstm_cell_10/ones_likeFill-lstm_10/lstm_cell_10/ones_like/Shape:output:0-lstm_10/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/ones_likeН
"lstm_10/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"lstm_10/lstm_cell_10/dropout/Const”
 lstm_10/lstm_cell_10/dropout/MulMul'lstm_10/lstm_cell_10/ones_like:output:0+lstm_10/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/lstm_cell_10/dropout/MulЯ
"lstm_10/lstm_cell_10/dropout/ShapeShape'lstm_10/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_10/lstm_cell_10/dropout/ShapeР
9lstm_10/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform+lstm_10/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ьщ±2;
9lstm_10/lstm_cell_10/dropout/random_uniform/RandomUniformЯ
+lstm_10/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+lstm_10/lstm_cell_10/dropout/GreaterEqual/yТ
)lstm_10/lstm_cell_10/dropout/GreaterEqualGreaterEqualBlstm_10/lstm_cell_10/dropout/random_uniform/RandomUniform:output:04lstm_10/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_10/lstm_cell_10/dropout/GreaterEqualЊ
!lstm_10/lstm_cell_10/dropout/CastCast-lstm_10/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_10/lstm_cell_10/dropout/Castќ
"lstm_10/lstm_cell_10/dropout/Mul_1Mul$lstm_10/lstm_cell_10/dropout/Mul:z:0%lstm_10/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/lstm_cell_10/dropout/Mul_1С
$lstm_10/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2&
$lstm_10/lstm_cell_10/dropout_1/Constў
"lstm_10/lstm_cell_10/dropout_1/MulMul'lstm_10/lstm_cell_10/ones_like:output:0-lstm_10/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/lstm_cell_10/dropout_1/Mul£
$lstm_10/lstm_cell_10/dropout_1/ShapeShape'lstm_10/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_10/lstm_cell_10/dropout_1/ShapeЦ
;lstm_10/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_10/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЊЮЄ2=
;lstm_10/lstm_cell_10/dropout_1/random_uniform/RandomUniform£
-lstm_10/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_10/lstm_cell_10/dropout_1/GreaterEqual/yЪ
+lstm_10/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualDlstm_10/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:06lstm_10/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_10/lstm_cell_10/dropout_1/GreaterEqualƒ
#lstm_10/lstm_cell_10/dropout_1/CastCast/lstm_10/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/lstm_cell_10/dropout_1/Cast÷
$lstm_10/lstm_cell_10/dropout_1/Mul_1Mul&lstm_10/lstm_cell_10/dropout_1/Mul:z:0'lstm_10/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/lstm_cell_10/dropout_1/Mul_1С
$lstm_10/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2&
$lstm_10/lstm_cell_10/dropout_2/Constў
"lstm_10/lstm_cell_10/dropout_2/MulMul'lstm_10/lstm_cell_10/ones_like:output:0-lstm_10/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/lstm_cell_10/dropout_2/Mul£
$lstm_10/lstm_cell_10/dropout_2/ShapeShape'lstm_10/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_10/lstm_cell_10/dropout_2/ShapeЦ
;lstm_10/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_10/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЦКЅ2=
;lstm_10/lstm_cell_10/dropout_2/random_uniform/RandomUniform£
-lstm_10/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_10/lstm_cell_10/dropout_2/GreaterEqual/yЪ
+lstm_10/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualDlstm_10/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:06lstm_10/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_10/lstm_cell_10/dropout_2/GreaterEqualƒ
#lstm_10/lstm_cell_10/dropout_2/CastCast/lstm_10/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/lstm_cell_10/dropout_2/Cast÷
$lstm_10/lstm_cell_10/dropout_2/Mul_1Mul&lstm_10/lstm_cell_10/dropout_2/Mul:z:0'lstm_10/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/lstm_cell_10/dropout_2/Mul_1С
$lstm_10/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2&
$lstm_10/lstm_cell_10/dropout_3/Constў
"lstm_10/lstm_cell_10/dropout_3/MulMul'lstm_10/lstm_cell_10/ones_like:output:0-lstm_10/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/lstm_cell_10/dropout_3/Mul£
$lstm_10/lstm_cell_10/dropout_3/ShapeShape'lstm_10/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_10/lstm_cell_10/dropout_3/ShapeЦ
;lstm_10/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_10/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЖВЃ2=
;lstm_10/lstm_cell_10/dropout_3/random_uniform/RandomUniform£
-lstm_10/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2/
-lstm_10/lstm_cell_10/dropout_3/GreaterEqual/yЪ
+lstm_10/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualDlstm_10/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:06lstm_10/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm_10/lstm_cell_10/dropout_3/GreaterEqualƒ
#lstm_10/lstm_cell_10/dropout_3/CastCast/lstm_10/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/lstm_cell_10/dropout_3/Cast÷
$lstm_10/lstm_cell_10/dropout_3/Mul_1Mul&lstm_10/lstm_cell_10/dropout_3/Mul:z:0'lstm_10/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/lstm_cell_10/dropout_3/Mul_1О
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dim 
)lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)lstm_10/lstm_cell_10/split/ReadVariableOpы
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:01lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_10/lstm_cell_10/splitљ
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMulЅ
lstm_10/lstm_cell_10/MatMul_1MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_1Ѕ
lstm_10/lstm_cell_10/MatMul_2MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_2Ѕ
lstm_10/lstm_cell_10/MatMul_3MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_3Т
&lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_10/lstm_cell_10/split_1/split_dimћ
+lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_10/lstm_cell_10/split_1/ReadVariableOpу
lstm_10/lstm_cell_10/split_1Split/lstm_10/lstm_cell_10/split_1/split_dim:output:03lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_10/lstm_cell_10/split_1«
lstm_10/lstm_cell_10/BiasAddBiasAdd%lstm_10/lstm_cell_10/MatMul:product:0%lstm_10/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/BiasAddЌ
lstm_10/lstm_cell_10/BiasAdd_1BiasAdd'lstm_10/lstm_cell_10/MatMul_1:product:0%lstm_10/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_1Ќ
lstm_10/lstm_cell_10/BiasAdd_2BiasAdd'lstm_10/lstm_cell_10/MatMul_2:product:0%lstm_10/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_2Ќ
lstm_10/lstm_cell_10/BiasAdd_3BiasAdd'lstm_10/lstm_cell_10/MatMul_3:product:0%lstm_10/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/BiasAdd_3≠
lstm_10/lstm_cell_10/mulMullstm_10/zeros:output:0&lstm_10/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul≥
lstm_10/lstm_cell_10/mul_1Mullstm_10/zeros:output:0(lstm_10/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_1≥
lstm_10/lstm_cell_10/mul_2Mullstm_10/zeros:output:0(lstm_10/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_2≥
lstm_10/lstm_cell_10/mul_3Mullstm_10/zeros:output:0(lstm_10/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_3Є
#lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_10/lstm_cell_10/ReadVariableOp•
(lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_10/lstm_cell_10/strided_slice/stack©
*lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_10/lstm_cell_10/strided_slice/stack_1©
*lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_10/lstm_cell_10/strided_slice/stack_2ъ
"lstm_10/lstm_cell_10/strided_sliceStridedSlice+lstm_10/lstm_cell_10/ReadVariableOp:value:01lstm_10/lstm_cell_10/strided_slice/stack:output:03lstm_10/lstm_cell_10/strided_slice/stack_1:output:03lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_10/lstm_cell_10/strided_slice≈
lstm_10/lstm_cell_10/MatMul_4MatMullstm_10/lstm_cell_10/mul:z:0+lstm_10/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_4њ
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/BiasAdd:output:0'lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/addЧ
lstm_10/lstm_cell_10/SigmoidSigmoidlstm_10/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/SigmoidЉ
%lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_1©
*lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_10/lstm_cell_10/strided_slice_1/stack≠
,lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_10/lstm_cell_10/strided_slice_1/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_1StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_1:value:03lstm_10/lstm_cell_10/strided_slice_1/stack:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_1…
lstm_10/lstm_cell_10/MatMul_5MatMullstm_10/lstm_cell_10/mul_1:z:0-lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_5≈
lstm_10/lstm_cell_10/add_1AddV2'lstm_10/lstm_cell_10/BiasAdd_1:output:0'lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_1Э
lstm_10/lstm_cell_10/Sigmoid_1Sigmoidlstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/Sigmoid_1ѓ
lstm_10/lstm_cell_10/mul_4Mul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_4Љ
%lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_2©
*lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_10/lstm_cell_10/strided_slice_2/stack≠
,lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_10/lstm_cell_10/strided_slice_2/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_2StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_2:value:03lstm_10/lstm_cell_10/strided_slice_2/stack:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_2…
lstm_10/lstm_cell_10/MatMul_6MatMullstm_10/lstm_cell_10/mul_2:z:0-lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_6≈
lstm_10/lstm_cell_10/add_2AddV2'lstm_10/lstm_cell_10/BiasAdd_2:output:0'lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_2Р
lstm_10/lstm_cell_10/ReluRelulstm_10/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/ReluЉ
lstm_10/lstm_cell_10/mul_5Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_5≥
lstm_10/lstm_cell_10/add_3AddV2lstm_10/lstm_cell_10/mul_4:z:0lstm_10/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_3Љ
%lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02'
%lstm_10/lstm_cell_10/ReadVariableOp_3©
*lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_10/lstm_cell_10/strided_slice_3/stack≠
,lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_10/lstm_cell_10/strided_slice_3/stack_1≠
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Ж
$lstm_10/lstm_cell_10/strided_slice_3StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_3:value:03lstm_10/lstm_cell_10/strided_slice_3/stack:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_10/lstm_cell_10/strided_slice_3…
lstm_10/lstm_cell_10/MatMul_7MatMullstm_10/lstm_cell_10/mul_3:z:0-lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/MatMul_7≈
lstm_10/lstm_cell_10/add_4AddV2'lstm_10/lstm_cell_10/BiasAdd_3:output:0'lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/add_4Э
lstm_10/lstm_cell_10/Sigmoid_2Sigmoidlstm_10/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/lstm_cell_10/Sigmoid_2Ф
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/Relu_1ј
lstm_10/lstm_cell_10/mul_6Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/lstm_cell_10/mul_6Я
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2'
%lstm_10/TensorArrayV2_1/element_shapeЎ
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2_1^
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/timeП
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counterщ
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_10_lstm_cell_10_split_readvariableop_resource4lstm_10_lstm_cell_10_split_1_readvariableop_resource,lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_10_while_body_437770*%
condR
lstm_10_while_cond_437769*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_10/while≈
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStackС
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_10/strided_slice_3/stackМ
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1М
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2 
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_10/strided_slice_3Й
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/perm≈
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_10/transpose_1v
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/runtime®
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_12/MatMul/ReadVariableOp®
dense_12/MatMulMatMul lstm_10/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_12/Relu®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/BiasAddk
reshape_6/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_6/ShapeИ
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stackМ
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1М
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2Ю
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_6/strided_slicex
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/1x
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/2“
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shape§
reshape_6/ReshapeReshapedense_13/BiasAdd:output:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_6/Reshapeт
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mul«
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/muly
IdentityIdentityreshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityќ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp0^dense_13/bias/Regularizer/Square/ReadVariableOp$^lstm_10/lstm_cell_10/ReadVariableOp&^lstm_10/lstm_cell_10/ReadVariableOp_1&^lstm_10/lstm_cell_10/ReadVariableOp_2&^lstm_10/lstm_cell_10/ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*^lstm_10/lstm_cell_10/split/ReadVariableOp,^lstm_10/lstm_cell_10/split_1/ReadVariableOp^lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2J
#lstm_10/lstm_cell_10/ReadVariableOp#lstm_10/lstm_cell_10/ReadVariableOp2N
%lstm_10/lstm_cell_10/ReadVariableOp_1%lstm_10/lstm_cell_10/ReadVariableOp_12N
%lstm_10/lstm_cell_10/ReadVariableOp_2%lstm_10/lstm_cell_10/ReadVariableOp_22N
%lstm_10/lstm_cell_10/ReadVariableOp_3%lstm_10/lstm_cell_10/ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_10/lstm_cell_10/split/ReadVariableOp)lstm_10/lstm_cell_10/split/ReadVariableOp2Z
+lstm_10/lstm_cell_10/split_1/ReadVariableOp+lstm_10/lstm_cell_10/split_1/ReadVariableOp2
lstm_10/whilelstm_10/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
№
 
__inference_loss_fn_1_439438Y
Flstm_10_lstm_cell_10_kernel_regularizer_square_readvariableop_resource:	А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЖ
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_10_lstm_cell_10_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muly
IdentityIdentity/lstm_10/lstm_cell_10/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityО
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
Ў%
г
while_body_436076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_10_436100_0:	А*
while_lstm_cell_10_436102_0:	А.
while_lstm_cell_10_436104_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_10_436100:	А(
while_lstm_cell_10_436102:	А,
while_lstm_cell_10_436104:	 АИҐ*while/lstm_cell_10/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemб
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_436100_0while_lstm_cell_10_436102_0while_lstm_cell_10_436104_0*
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4359982,
*while/lstm_cell_10/StatefulPartitionedCallч
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/Identity_3§
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4§
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_436100while_lstm_cell_10_436100_0"8
while_lstm_cell_10_436102while_lstm_cell_10_436102_0"8
while_lstm_cell_10_436104while_lstm_cell_10_436104_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
а	
¶
-__inference_sequential_4_layer_call_fn_437357

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallљ
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_4371762
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆

г
lstm_10_while_cond_437466,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1D
@lstm_10_while_lstm_10_while_cond_437466___redundant_placeholder0D
@lstm_10_while_lstm_10_while_cond_437466___redundant_placeholder1D
@lstm_10_while_lstm_10_while_cond_437466___redundant_placeholder2D
@lstm_10_while_lstm_10_while_cond_437466___redundant_placeholder3
lstm_10_while_identity
Ш
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2
lstm_10/while/Lessu
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_10/while/Identity"9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
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
¶
µ
(__inference_lstm_10_layer_call_fn_438013

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallА
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4371122
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
џ+
Ђ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437246
input_5!
lstm_10_437215:	А
lstm_10_437217:	А!
lstm_10_437219:	 А!
dense_12_437222:  
dense_12_437224: !
dense_13_437227: 
dense_13_437229:
identityИҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐlstm_10/StatefulPartitionedCallҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐ
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_10_437215lstm_10_437217lstm_10_437219*
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4366742!
lstm_10/StatefulPartitionedCallґ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_12_437222dense_12_437224*
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
GPU 2J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4366932"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_437227dense_13_437229*
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
GPU 2J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4367152"
 dense_13/StatefulPartitionedCallю
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_4367342
reshape_6/PartitionedCallќ
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_10_437215*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mulЃ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_437229*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulБ
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identity®
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall0^dense_13/bias/Regularizer/Square/ReadVariableOp ^lstm_10/StatefulPartitionedCall>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
м
І
D__inference_dense_13_layer_call_and_return_conditional_losses_436715

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ/dense_13/bias/Regularizer/Square/ReadVariableOpН
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
BiasAddЊ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_13/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
с
Ц
)__inference_dense_13_layer_call_fn_439148

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallф
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
GPU 2J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4367152
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
ё°
І
C__inference_lstm_10_layer_call_and_return_conditional_losses_438806

inputs=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3О
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulТ
lstm_cell_10/mul_1Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1Т
lstm_cell_10/mul_2Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2Т
lstm_cell_10/mul_3Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_438673*
condR
while_cond_438672*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_438947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_438947___redundant_placeholder04
0while_while_cond_438947___redundant_placeholder14
0while_while_cond_438947___redundant_placeholder24
0while_while_cond_438947___redundant_placeholder3
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
ґ
ц
-__inference_lstm_cell_10_layer_call_fn_439233

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall√
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4359982
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
’
√
while_cond_436946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_436946___redundant_placeholder04
0while_while_cond_436946___redundant_placeholder14
0while_while_cond_436946___redundant_placeholder24
0while_while_cond_436946___redundant_placeholder3
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
ґ
ц
-__inference_lstm_cell_10_layer_call_fn_439216

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall√
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4357652
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
ё°
І
C__inference_lstm_10_layer_call_and_return_conditional_losses_436674

inputs=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3О
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulТ
lstm_cell_10/mul_1Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1Т
lstm_cell_10/mul_2Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2Т
lstm_cell_10/mul_3Mulzeros:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_436541*
condR
while_cond_436540*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Їъ
м
!__inference__wrapped_model_435641
input_5R
?sequential_4_lstm_10_lstm_cell_10_split_readvariableop_resource:	АP
Asequential_4_lstm_10_lstm_cell_10_split_1_readvariableop_resource:	АL
9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource:	 АF
4sequential_4_dense_12_matmul_readvariableop_resource:  C
5sequential_4_dense_12_biasadd_readvariableop_resource: F
4sequential_4_dense_13_matmul_readvariableop_resource: C
5sequential_4_dense_13_biasadd_readvariableop_resource:
identityИҐ,sequential_4/dense_12/BiasAdd/ReadVariableOpҐ+sequential_4/dense_12/MatMul/ReadVariableOpҐ,sequential_4/dense_13/BiasAdd/ReadVariableOpҐ+sequential_4/dense_13/MatMul/ReadVariableOpҐ0sequential_4/lstm_10/lstm_cell_10/ReadVariableOpҐ2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_1Ґ2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_2Ґ2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_3Ґ6sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOpҐ8sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOpҐsequential_4/lstm_10/whileo
sequential_4/lstm_10/ShapeShapeinput_5*
T0*
_output_shapes
:2
sequential_4/lstm_10/ShapeЮ
(sequential_4/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/lstm_10/strided_slice/stackҐ
*sequential_4/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_10/strided_slice/stack_1Ґ
*sequential_4/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_10/strided_slice/stack_2а
"sequential_4/lstm_10/strided_sliceStridedSlice#sequential_4/lstm_10/Shape:output:01sequential_4/lstm_10/strided_slice/stack:output:03sequential_4/lstm_10/strided_slice/stack_1:output:03sequential_4/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_4/lstm_10/strided_sliceЖ
 sequential_4/lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_4/lstm_10/zeros/mul/yј
sequential_4/lstm_10/zeros/mulMul+sequential_4/lstm_10/strided_slice:output:0)sequential_4/lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_10/zeros/mulЙ
!sequential_4/lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2#
!sequential_4/lstm_10/zeros/Less/yї
sequential_4/lstm_10/zeros/LessLess"sequential_4/lstm_10/zeros/mul:z:0*sequential_4/lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_10/zeros/LessМ
#sequential_4/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_4/lstm_10/zeros/packed/1„
!sequential_4/lstm_10/zeros/packedPack+sequential_4/lstm_10/strided_slice:output:0,sequential_4/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_4/lstm_10/zeros/packedЙ
 sequential_4/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_4/lstm_10/zeros/Const…
sequential_4/lstm_10/zerosFill*sequential_4/lstm_10/zeros/packed:output:0)sequential_4/lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_4/lstm_10/zerosК
"sequential_4/lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_4/lstm_10/zeros_1/mul/y∆
 sequential_4/lstm_10/zeros_1/mulMul+sequential_4/lstm_10/strided_slice:output:0+sequential_4/lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_10/zeros_1/mulН
#sequential_4/lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2%
#sequential_4/lstm_10/zeros_1/Less/y√
!sequential_4/lstm_10/zeros_1/LessLess$sequential_4/lstm_10/zeros_1/mul:z:0,sequential_4/lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_4/lstm_10/zeros_1/LessР
%sequential_4/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_4/lstm_10/zeros_1/packed/1Ё
#sequential_4/lstm_10/zeros_1/packedPack+sequential_4/lstm_10/strided_slice:output:0.sequential_4/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/lstm_10/zeros_1/packedН
"sequential_4/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_4/lstm_10/zeros_1/Const—
sequential_4/lstm_10/zeros_1Fill,sequential_4/lstm_10/zeros_1/packed:output:0+sequential_4/lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_4/lstm_10/zeros_1Я
#sequential_4/lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_4/lstm_10/transpose/permЇ
sequential_4/lstm_10/transpose	Transposeinput_5,sequential_4/lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 
sequential_4/lstm_10/transposeО
sequential_4/lstm_10/Shape_1Shape"sequential_4/lstm_10/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_10/Shape_1Ґ
*sequential_4/lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_10/strided_slice_1/stack¶
,sequential_4/lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_10/strided_slice_1/stack_1¶
,sequential_4/lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_10/strided_slice_1/stack_2м
$sequential_4/lstm_10/strided_slice_1StridedSlice%sequential_4/lstm_10/Shape_1:output:03sequential_4/lstm_10/strided_slice_1/stack:output:05sequential_4/lstm_10/strided_slice_1/stack_1:output:05sequential_4/lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/lstm_10/strided_slice_1ѓ
0sequential_4/lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€22
0sequential_4/lstm_10/TensorArrayV2/element_shapeЖ
"sequential_4/lstm_10/TensorArrayV2TensorListReserve9sequential_4/lstm_10/TensorArrayV2/element_shape:output:0-sequential_4/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_4/lstm_10/TensorArrayV2й
Jsequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2L
Jsequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeћ
<sequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_10/transpose:y:0Ssequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensorҐ
*sequential_4/lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_10/strided_slice_2/stack¶
,sequential_4/lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_10/strided_slice_2/stack_1¶
,sequential_4/lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_10/strided_slice_2/stack_2ъ
$sequential_4/lstm_10/strided_slice_2StridedSlice"sequential_4/lstm_10/transpose:y:03sequential_4/lstm_10/strided_slice_2/stack:output:05sequential_4/lstm_10/strided_slice_2/stack_1:output:05sequential_4/lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2&
$sequential_4/lstm_10/strided_slice_2є
1sequential_4/lstm_10/lstm_cell_10/ones_like/ShapeShape#sequential_4/lstm_10/zeros:output:0*
T0*
_output_shapes
:23
1sequential_4/lstm_10/lstm_cell_10/ones_like/ShapeЂ
1sequential_4/lstm_10/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1sequential_4/lstm_10/lstm_cell_10/ones_like/ConstМ
+sequential_4/lstm_10/lstm_cell_10/ones_likeFill:sequential_4/lstm_10/lstm_cell_10/ones_like/Shape:output:0:sequential_4/lstm_10/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/ones_like®
1sequential_4/lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/lstm_10/lstm_cell_10/split/split_dimс
6sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp?sequential_4_lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype028
6sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOpѓ
'sequential_4/lstm_10/lstm_cell_10/splitSplit:sequential_4/lstm_10/lstm_cell_10/split/split_dim:output:0>sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2)
'sequential_4/lstm_10/lstm_cell_10/splitс
(sequential_4/lstm_10/lstm_cell_10/MatMulMatMul-sequential_4/lstm_10/strided_slice_2:output:00sequential_4/lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_4/lstm_10/lstm_cell_10/MatMulх
*sequential_4/lstm_10/lstm_cell_10/MatMul_1MatMul-sequential_4/lstm_10/strided_slice_2:output:00sequential_4/lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_1х
*sequential_4/lstm_10/lstm_cell_10/MatMul_2MatMul-sequential_4/lstm_10/strided_slice_2:output:00sequential_4/lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_2х
*sequential_4/lstm_10/lstm_cell_10/MatMul_3MatMul-sequential_4/lstm_10/strided_slice_2:output:00sequential_4/lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_3ђ
3sequential_4/lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_4/lstm_10/lstm_cell_10/split_1/split_dimу
8sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOpAsequential_4_lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02:
8sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOpІ
)sequential_4/lstm_10/lstm_cell_10/split_1Split<sequential_4/lstm_10/lstm_cell_10/split_1/split_dim:output:0@sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2+
)sequential_4/lstm_10/lstm_cell_10/split_1ы
)sequential_4/lstm_10/lstm_cell_10/BiasAddBiasAdd2sequential_4/lstm_10/lstm_cell_10/MatMul:product:02sequential_4/lstm_10/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_4/lstm_10/lstm_cell_10/BiasAddБ
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_1BiasAdd4sequential_4/lstm_10/lstm_cell_10/MatMul_1:product:02sequential_4/lstm_10/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_1Б
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_2BiasAdd4sequential_4/lstm_10/lstm_cell_10/MatMul_2:product:02sequential_4/lstm_10/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_2Б
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_3BiasAdd4sequential_4/lstm_10/lstm_cell_10/MatMul_3:product:02sequential_4/lstm_10/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/BiasAdd_3в
%sequential_4/lstm_10/lstm_cell_10/mulMul#sequential_4/lstm_10/zeros:output:04sequential_4/lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_4/lstm_10/lstm_cell_10/mulж
'sequential_4/lstm_10/lstm_cell_10/mul_1Mul#sequential_4/lstm_10/zeros:output:04sequential_4/lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_1ж
'sequential_4/lstm_10/lstm_cell_10/mul_2Mul#sequential_4/lstm_10/zeros:output:04sequential_4/lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_2ж
'sequential_4/lstm_10/lstm_cell_10/mul_3Mul#sequential_4/lstm_10/zeros:output:04sequential_4/lstm_10/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_3я
0sequential_4/lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype022
0sequential_4/lstm_10/lstm_cell_10/ReadVariableOpњ
5sequential_4/lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_4/lstm_10/lstm_cell_10/strided_slice/stack√
7sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_1√
7sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_2»
/sequential_4/lstm_10/lstm_cell_10/strided_sliceStridedSlice8sequential_4/lstm_10/lstm_cell_10/ReadVariableOp:value:0>sequential_4/lstm_10/lstm_cell_10/strided_slice/stack:output:0@sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_1:output:0@sequential_4/lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_4/lstm_10/lstm_cell_10/strided_sliceщ
*sequential_4/lstm_10/lstm_cell_10/MatMul_4MatMul)sequential_4/lstm_10/lstm_cell_10/mul:z:08sequential_4/lstm_10/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_4у
%sequential_4/lstm_10/lstm_cell_10/addAddV22sequential_4/lstm_10/lstm_cell_10/BiasAdd:output:04sequential_4/lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_4/lstm_10/lstm_cell_10/addЊ
)sequential_4/lstm_10/lstm_cell_10/SigmoidSigmoid)sequential_4/lstm_10/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_4/lstm_10/lstm_cell_10/Sigmoidг
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype024
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_1√
7sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_1«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_2‘
1sequential_4/lstm_10/lstm_cell_10/strided_slice_1StridedSlice:sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_1:value:0@sequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_4/lstm_10/lstm_cell_10/strided_slice_1э
*sequential_4/lstm_10/lstm_cell_10/MatMul_5MatMul+sequential_4/lstm_10/lstm_cell_10/mul_1:z:0:sequential_4/lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_5щ
'sequential_4/lstm_10/lstm_cell_10/add_1AddV24sequential_4/lstm_10/lstm_cell_10/BiasAdd_1:output:04sequential_4/lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/add_1ƒ
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_1Sigmoid+sequential_4/lstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_1г
'sequential_4/lstm_10/lstm_cell_10/mul_4Mul/sequential_4/lstm_10/lstm_cell_10/Sigmoid_1:y:0%sequential_4/lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_4г
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype024
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_2√
7sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_1«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_2‘
1sequential_4/lstm_10/lstm_cell_10/strided_slice_2StridedSlice:sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_2:value:0@sequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_4/lstm_10/lstm_cell_10/strided_slice_2э
*sequential_4/lstm_10/lstm_cell_10/MatMul_6MatMul+sequential_4/lstm_10/lstm_cell_10/mul_2:z:0:sequential_4/lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_6щ
'sequential_4/lstm_10/lstm_cell_10/add_2AddV24sequential_4/lstm_10/lstm_cell_10/BiasAdd_2:output:04sequential_4/lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/add_2Ј
&sequential_4/lstm_10/lstm_cell_10/ReluRelu+sequential_4/lstm_10/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_4/lstm_10/lstm_cell_10/Reluр
'sequential_4/lstm_10/lstm_cell_10/mul_5Mul-sequential_4/lstm_10/lstm_cell_10/Sigmoid:y:04sequential_4/lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_5з
'sequential_4/lstm_10/lstm_cell_10/add_3AddV2+sequential_4/lstm_10/lstm_cell_10/mul_4:z:0+sequential_4/lstm_10/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/add_3г
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype024
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_3√
7sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   29
7sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_1«
9sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_2‘
1sequential_4/lstm_10/lstm_cell_10/strided_slice_3StridedSlice:sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_3:value:0@sequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:0Bsequential_4/lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_4/lstm_10/lstm_cell_10/strided_slice_3э
*sequential_4/lstm_10/lstm_cell_10/MatMul_7MatMul+sequential_4/lstm_10/lstm_cell_10/mul_3:z:0:sequential_4/lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_4/lstm_10/lstm_cell_10/MatMul_7щ
'sequential_4/lstm_10/lstm_cell_10/add_4AddV24sequential_4/lstm_10/lstm_cell_10/BiasAdd_3:output:04sequential_4/lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/add_4ƒ
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_2Sigmoid+sequential_4/lstm_10/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_2ї
(sequential_4/lstm_10/lstm_cell_10/Relu_1Relu+sequential_4/lstm_10/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_4/lstm_10/lstm_cell_10/Relu_1ф
'sequential_4/lstm_10/lstm_cell_10/mul_6Mul/sequential_4/lstm_10/lstm_cell_10/Sigmoid_2:y:06sequential_4/lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_4/lstm_10/lstm_cell_10/mul_6є
2sequential_4/lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    24
2sequential_4/lstm_10/TensorArrayV2_1/element_shapeМ
$sequential_4/lstm_10/TensorArrayV2_1TensorListReserve;sequential_4/lstm_10/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_4/lstm_10/TensorArrayV2_1x
sequential_4/lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_10/time©
-sequential_4/lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-sequential_4/lstm_10/while/maximum_iterationsФ
'sequential_4/lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_4/lstm_10/while/loop_counterЉ
sequential_4/lstm_10/whileWhile0sequential_4/lstm_10/while/loop_counter:output:06sequential_4/lstm_10/while/maximum_iterations:output:0"sequential_4/lstm_10/time:output:0-sequential_4/lstm_10/TensorArrayV2_1:handle:0#sequential_4/lstm_10/zeros:output:0%sequential_4/lstm_10/zeros_1:output:0-sequential_4/lstm_10/strided_slice_1:output:0Lsequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_4_lstm_10_lstm_cell_10_split_readvariableop_resourceAsequential_4_lstm_10_lstm_cell_10_split_1_readvariableop_resource9sequential_4_lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_10_while_body_435492*2
cond*R(
&sequential_4_lstm_10_while_cond_435491*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
sequential_4/lstm_10/whileя
Esequential_4/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2G
Esequential_4/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeЉ
7sequential_4/lstm_10/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_10/while:output:3Nsequential_4/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype029
7sequential_4/lstm_10/TensorArrayV2Stack/TensorListStackЂ
*sequential_4/lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2,
*sequential_4/lstm_10/strided_slice_3/stack¶
,sequential_4/lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_4/lstm_10/strided_slice_3/stack_1¶
,sequential_4/lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_10/strided_slice_3/stack_2Ш
$sequential_4/lstm_10/strided_slice_3StridedSlice@sequential_4/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_10/strided_slice_3/stack:output:05sequential_4/lstm_10/strided_slice_3/stack_1:output:05sequential_4/lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2&
$sequential_4/lstm_10/strided_slice_3£
%sequential_4/lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_4/lstm_10/transpose_1/permщ
 sequential_4/lstm_10/transpose_1	Transpose@sequential_4/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2"
 sequential_4/lstm_10/transpose_1Р
sequential_4/lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_10/runtimeѕ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp№
sequential_4/dense_12/MatMulMatMul-sequential_4/lstm_10/strided_slice_3:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_4/dense_12/MatMulќ
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpў
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_4/dense_12/BiasAddЪ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_4/dense_12/Reluѕ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp„
sequential_4/dense_13/MatMulMatMul(sequential_4/dense_12/Relu:activations:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_4/dense_13/MatMulќ
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpў
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_4/dense_13/BiasAddТ
sequential_4/reshape_6/ShapeShape&sequential_4/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_4/reshape_6/ShapeҐ
*sequential_4/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_6/strided_slice/stack¶
,sequential_4/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_6/strided_slice/stack_1¶
,sequential_4/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_6/strided_slice/stack_2м
$sequential_4/reshape_6/strided_sliceStridedSlice%sequential_4/reshape_6/Shape:output:03sequential_4/reshape_6/strided_slice/stack:output:05sequential_4/reshape_6/strided_slice/stack_1:output:05sequential_4/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_6/strided_sliceТ
&sequential_4/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_6/Reshape/shape/1Т
&sequential_4/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_6/Reshape/shape/2У
$sequential_4/reshape_6/Reshape/shapePack-sequential_4/reshape_6/strided_slice:output:0/sequential_4/reshape_6/Reshape/shape/1:output:0/sequential_4/reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_6/Reshape/shapeЎ
sequential_4/reshape_6/ReshapeReshape&sequential_4/dense_13/BiasAdd:output:0-sequential_4/reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 
sequential_4/reshape_6/ReshapeЖ
IdentityIdentity'sequential_4/reshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityл
NoOpNoOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp1^sequential_4/lstm_10/lstm_cell_10/ReadVariableOp3^sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_13^sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_23^sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_37^sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOp9^sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOp^sequential_4/lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp2d
0sequential_4/lstm_10/lstm_cell_10/ReadVariableOp0sequential_4/lstm_10/lstm_cell_10/ReadVariableOp2h
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_12sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_12h
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_22sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_22h
2sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_32sequential_4/lstm_10/lstm_cell_10/ReadVariableOp_32p
6sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOp6sequential_4/lstm_10/lstm_cell_10/split/ReadVariableOp2t
8sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOp8sequential_4/lstm_10/lstm_cell_10/split_1/ReadVariableOp28
sequential_4/lstm_10/whilesequential_4/lstm_10/while:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
Њ
Ј
(__inference_lstm_10_layer_call_fn_437980
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallВ
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4358542
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Іv
й
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_435998

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
seed2ГєЦ2&
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
seed2фъМ2(
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
seed2Аф∞2(
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
seed2≠ѕи2(
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6Ё
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muld
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

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
ьФ
Љ
lstm_10_while_body_437467,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	АK
<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	АG
4lstm_10_while_lstm_cell_10_readvariableop_resource_0:	 А
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorK
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:	АI
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	АE
2lstm_10_while_lstm_cell_10_readvariableop_resource:	 АИҐ)lstm_10/while/lstm_cell_10/ReadVariableOpҐ+lstm_10/while/lstm_cell_10/ReadVariableOp_1Ґ+lstm_10/while/lstm_cell_10/ReadVariableOp_2Ґ+lstm_10/while/lstm_cell_10/ReadVariableOp_3Ґ/lstm_10/while/lstm_cell_10/split/ReadVariableOpҐ1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp”
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItem£
*lstm_10/while/lstm_cell_10/ones_like/ShapeShapelstm_10_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_10/while/lstm_cell_10/ones_like/ShapeЭ
*lstm_10/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*lstm_10/while/lstm_cell_10/ones_like/Constр
$lstm_10/while/lstm_cell_10/ones_likeFill3lstm_10/while/lstm_cell_10/ones_like/Shape:output:03lstm_10/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/ones_likeЪ
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dimё
/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOp:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype021
/lstm_10/while/lstm_cell_10/split/ReadVariableOpУ
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:07lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_10/while/lstm_cell_10/splitз
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_10/while/lstm_cell_10/MatMulл
#lstm_10/while/lstm_cell_10/MatMul_1MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_1л
#lstm_10/while/lstm_cell_10/MatMul_2MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_2л
#lstm_10/while/lstm_cell_10/MatMul_3MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_3Ю
,lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_10/while/lstm_cell_10/split_1/split_dimа
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpЛ
"lstm_10/while/lstm_cell_10/split_1Split5lstm_10/while/lstm_cell_10/split_1/split_dim:output:09lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_10/while/lstm_cell_10/split_1я
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd+lstm_10/while/lstm_cell_10/MatMul:product:0+lstm_10/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/while/lstm_cell_10/BiasAddе
$lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd-lstm_10/while/lstm_cell_10/MatMul_1:product:0+lstm_10/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_1е
$lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd-lstm_10/while/lstm_cell_10/MatMul_2:product:0+lstm_10/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_2е
$lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd-lstm_10/while/lstm_cell_10/MatMul_3:product:0+lstm_10/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/BiasAdd_3≈
lstm_10/while/lstm_cell_10/mulMullstm_10_while_placeholder_2-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/while/lstm_cell_10/mul…
 lstm_10/while/lstm_cell_10/mul_1Mullstm_10_while_placeholder_2-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_1…
 lstm_10/while/lstm_cell_10/mul_2Mullstm_10_while_placeholder_2-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_2…
 lstm_10/while/lstm_cell_10/mul_3Mullstm_10_while_placeholder_2-lstm_10/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_3ћ
)lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_10/while/lstm_cell_10/ReadVariableOp±
.lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_10/while/lstm_cell_10/strided_slice/stackµ
0lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_10/while/lstm_cell_10/strided_slice/stack_1µ
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Ю
(lstm_10/while/lstm_cell_10/strided_sliceStridedSlice1lstm_10/while/lstm_cell_10/ReadVariableOp:value:07lstm_10/while/lstm_cell_10/strided_slice/stack:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_10/while/lstm_cell_10/strided_sliceЁ
#lstm_10/while/lstm_cell_10/MatMul_4MatMul"lstm_10/while/lstm_cell_10/mul:z:01lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_4„
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/BiasAdd:output:0-lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_10/while/lstm_cell_10/add©
"lstm_10/while/lstm_cell_10/SigmoidSigmoid"lstm_10/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_10/while/lstm_cell_10/Sigmoid–
+lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_1µ
0lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_10/while/lstm_cell_10/strided_slice_1/stackє
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_1StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:09lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_1б
#lstm_10/while/lstm_cell_10/MatMul_5MatMul$lstm_10/while/lstm_cell_10/mul_1:z:03lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_5Ё
 lstm_10/while/lstm_cell_10/add_1AddV2-lstm_10/while/lstm_cell_10/BiasAdd_1:output:0-lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_1ѓ
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/Sigmoid_1ƒ
 lstm_10/while/lstm_cell_10/mul_4Mul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_4–
+lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_2µ
0lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_10/while/lstm_cell_10/strided_slice_2/stackє
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_2StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:09lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_2б
#lstm_10/while/lstm_cell_10/MatMul_6MatMul$lstm_10/while/lstm_cell_10/mul_2:z:03lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_6Ё
 lstm_10/while/lstm_cell_10/add_2AddV2-lstm_10/while/lstm_cell_10/BiasAdd_2:output:0-lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_2Ґ
lstm_10/while/lstm_cell_10/ReluRelu$lstm_10/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_10/while/lstm_cell_10/Relu‘
 lstm_10/while/lstm_cell_10/mul_5Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_5Ћ
 lstm_10/while/lstm_cell_10/add_3AddV2$lstm_10/while/lstm_cell_10/mul_4:z:0$lstm_10/while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_3–
+lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02-
+lstm_10/while/lstm_cell_10/ReadVariableOp_3µ
0lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_10/while/lstm_cell_10/strided_slice_3/stackє
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1є
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2™
*lstm_10/while/lstm_cell_10/strided_slice_3StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:09lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_10/while/lstm_cell_10/strided_slice_3б
#lstm_10/while/lstm_cell_10/MatMul_7MatMul$lstm_10/while/lstm_cell_10/mul_3:z:03lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_10/while/lstm_cell_10/MatMul_7Ё
 lstm_10/while/lstm_cell_10/add_4AddV2-lstm_10/while/lstm_cell_10/BiasAdd_3:output:0-lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/add_4ѓ
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid$lstm_10/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_10/while/lstm_cell_10/Sigmoid_2¶
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_10/while/lstm_cell_10/Relu_1Ў
 lstm_10/while/lstm_cell_10/mul_6Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_10/while/lstm_cell_10/mul_6И
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_10/while/TensorArrayV2Write/TensorListSetIteml
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add/yЙ
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/addp
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add_1/yЮ
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1Л
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity¶
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1Н
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2Ї
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3≠
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_6:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/while/Identity_4≠
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_3:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_10/while/Identity_5Ж
lstm_10/while/NoOpNoOp*^lstm_10/while/lstm_cell_10/ReadVariableOp,^lstm_10/while/lstm_cell_10/ReadVariableOp_1,^lstm_10/while/lstm_cell_10/ReadVariableOp_2,^lstm_10/while/lstm_cell_10/ReadVariableOp_30^lstm_10/while/lstm_cell_10/split/ReadVariableOp2^lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_10/while/NoOp"9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"j
2lstm_10_while_lstm_cell_10_readvariableop_resource4lstm_10_while_lstm_cell_10_readvariableop_resource_0"z
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"v
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"»
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)lstm_10/while/lstm_cell_10/ReadVariableOp)lstm_10/while/lstm_cell_10/ReadVariableOp2Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_1+lstm_10/while/lstm_cell_10/ReadVariableOp_12Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_2+lstm_10/while/lstm_cell_10/ReadVariableOp_22Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_3+lstm_10/while/lstm_cell_10/ReadVariableOp_32b
/lstm_10/while/lstm_cell_10/split/ReadVariableOp/lstm_10/while/lstm_cell_10/split/ReadVariableOp2f
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp: 
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
О–
©
C__inference_lstm_10_layer_call_and_return_conditional_losses_438563
inputs_0=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout/Const≥
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/MulЗ
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeч
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2вЬU23
1lstm_cell_10/dropout/random_uniform/RandomUniformП
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_10/dropout/GreaterEqual/yт
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_cell_10/dropout/GreaterEqual¶
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/CastЃ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/Mul_1Б
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_1/Constє
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/MulЛ
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shapeю
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2±щН25
3lstm_cell_10/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_1/GreaterEqual/yъ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_1/GreaterEqualђ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Castґ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Mul_1Б
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_2/Constє
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/MulЛ
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shapeю
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Яъъ25
3lstm_cell_10/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_2/GreaterEqual/yъ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_2/GreaterEqualђ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Castґ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Mul_1Б
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_3/Constє
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/MulЛ
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shapeю
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2уі«25
3lstm_cell_10/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_3/GreaterEqual/yъ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_3/GreaterEqualђ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Castґ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3Н
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulУ
lstm_cell_10/mul_1Mulzeros:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1У
lstm_cell_10/mul_2Mulzeros:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2У
lstm_cell_10/mul_3Mulzeros:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_438398*
condR
while_cond_438397*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
В
х
D__inference_dense_12_layer_call_and_return_conditional_losses_439133

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
Ў+
™
H__inference_sequential_4_layer_call_and_return_conditional_losses_437176

inputs!
lstm_10_437145:	А
lstm_10_437147:	А!
lstm_10_437149:	 А!
dense_12_437152:  
dense_12_437154: !
dense_13_437157: 
dense_13_437159:
identityИҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ/dense_13/bias/Regularizer/Square/ReadVariableOpҐlstm_10/StatefulPartitionedCallҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp°
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_437145lstm_10_437147lstm_10_437149*
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4371122!
lstm_10/StatefulPartitionedCallґ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_12_437152dense_12_437154*
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
GPU 2J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4366932"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_437157dense_13_437159*
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
GPU 2J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4367152"
 dense_13/StatefulPartitionedCallю
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_4367342
reshape_6/PartitionedCallќ
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_10_437145*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/mulЃ
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_437159*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulБ
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identity®
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall0^dense_13/bias/Regularizer/Square/ReadVariableOp ^lstm_10/StatefulPartitionedCall>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ІА
§	
while_body_438673
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3•
while/lstm_cell_10/mulMulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul©
while/lstm_cell_10/mul_1Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1©
while/lstm_cell_10/mul_2Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2©
while/lstm_cell_10/mul_3Mulwhile_placeholder_2%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
ц
©
__inference_loss_fn_0_439193F
8dense_13_bias_regularizer_square_readvariableop_resource:
identityИҐ/dense_13/bias/Regularizer/Square/ReadVariableOp„
/dense_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_13_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_13/bias/Regularizer/Square/ReadVariableOpђ
 dense_13/bias/Regularizer/SquareSquare7dense_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_13/bias/Regularizer/SquareМ
dense_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_13/bias/Regularizer/Constґ
dense_13/bias/Regularizer/SumSum$dense_13/bias/Regularizer/Square:y:0(dense_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/SumЗ
dense_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_13/bias/Regularizer/mul/xЄ
dense_13/bias/Regularizer/mulMul(dense_13/bias/Regularizer/mul/x:output:0&dense_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_13/bias/Regularizer/mulk
IdentityIdentity!dense_13/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityА
NoOpNoOp0^dense_13/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_13/bias/Regularizer/Square/ReadVariableOp/dense_13/bias/Regularizer/Square/ReadVariableOp
ёR
л
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439314

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6Ё
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muld
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

Identity_2И
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
а	
¶
-__inference_sequential_4_layer_call_fn_437338

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallљ
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_4367492
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы≤
§	
while_body_438398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeЙ
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2"
 while/lstm_cell_10/dropout/ConstЋ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_10/dropout/MulЩ
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/ShapeК
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЂЯи29
7while/lstm_cell_10/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_10/dropout/GreaterEqual/yК
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'while/lstm_cell_10/dropout/GreaterEqualЄ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_10/dropout/Cast∆
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout/Mul_1Н
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_1/Const—
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_1/MulЭ
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/ShapeР
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ШиЛ2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/yТ
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_1/GreaterEqualЊ
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_1/Castќ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_1/Mul_1Н
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_2/Const—
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_2/MulЭ
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/ShapeР
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2т£ƒ2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/yТ
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_2/GreaterEqualЊ
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_2/Castќ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_2/Mul_1Н
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_3/Const—
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_3/MulЭ
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/ShapeР
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2евЊ2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/yТ
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_3/GreaterEqualЊ
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_3/Castќ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_3/Mul_1К
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3§
while/lstm_cell_10/mulMulwhile_placeholder_2$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul™
while/lstm_cell_10/mul_1Mulwhile_placeholder_2&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1™
while/lstm_cell_10/mul_2Mulwhile_placeholder_2&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2™
while/lstm_cell_10/mul_3Mulwhile_placeholder_2&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
Ў%
г
while_body_435779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_10_435803_0:	А*
while_lstm_cell_10_435805_0:	А.
while_lstm_cell_10_435807_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_10_435803:	А(
while_lstm_cell_10_435805:	А,
while_lstm_cell_10_435807:	 АИҐ*while/lstm_cell_10/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemб
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_435803_0while_lstm_cell_10_435805_0while_lstm_cell_10_435807_0*
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
GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_4357652,
*while/lstm_cell_10/StatefulPartitionedCallч
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/Identity_3§
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4§
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_435803while_lstm_cell_10_435803_0"8
while_lstm_cell_10_435805while_lstm_cell_10_435805_0"8
while_lstm_cell_10_435807while_lstm_cell_10_435807_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
Њ
Ј
(__inference_lstm_10_layer_call_fn_437991
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallВ
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
GPU 2J 8В *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_4361512
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
°|
А
"__inference__traced_restore_439639
file_prefix2
 assignvariableop_dense_12_kernel:  .
 assignvariableop_1_dense_12_bias: 4
"assignvariableop_2_dense_13_kernel: .
 assignvariableop_3_dense_13_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_10_lstm_cell_10_kernel:	АL
9assignvariableop_10_lstm_10_lstm_cell_10_recurrent_kernel:	 А<
-assignvariableop_11_lstm_10_lstm_cell_10_bias:	А#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_12_kernel_m:  6
(assignvariableop_15_adam_dense_12_bias_m: <
*assignvariableop_16_adam_dense_13_kernel_m: 6
(assignvariableop_17_adam_dense_13_bias_m:I
6assignvariableop_18_adam_lstm_10_lstm_cell_10_kernel_m:	АS
@assignvariableop_19_adam_lstm_10_lstm_cell_10_recurrent_kernel_m:	 АC
4assignvariableop_20_adam_lstm_10_lstm_cell_10_bias_m:	А<
*assignvariableop_21_adam_dense_12_kernel_v:  6
(assignvariableop_22_adam_dense_12_bias_v: <
*assignvariableop_23_adam_dense_13_kernel_v: 6
(assignvariableop_24_adam_dense_13_bias_v:I
6assignvariableop_25_adam_lstm_10_lstm_cell_10_kernel_v:	АS
@assignvariableop_26_adam_lstm_10_lstm_cell_10_recurrent_kernel_v:	 АC
4assignvariableop_27_adam_lstm_10_lstm_cell_10_bias_v:	А
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

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9≥
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_10_lstm_cell_10_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_10_lstm_cell_10_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_10_lstm_cell_10_biasIdentity_11:output:0"/device:CPU:0*
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
Identity_14≤
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_12_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15∞
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_12_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16≤
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_13_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17∞
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_13_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Њ
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_10_lstm_cell_10_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19»
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_10_lstm_cell_10_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Љ
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_10_lstm_cell_10_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21≤
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_12_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22∞
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_12_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≤
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_13_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24∞
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_13_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Њ
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_10_lstm_cell_10_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26»
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_10_lstm_cell_10_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Љ
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_10_lstm_cell_10_bias_vIdentity_27:output:0"/device:CPU:0*
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
ъ≤
§	
while_body_438948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeЙ
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2"
 while/lstm_cell_10/dropout/ConstЋ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_10/dropout/MulЩ
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/ShapeК
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2¬—о29
7while/lstm_cell_10/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_10/dropout/GreaterEqual/yК
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'while/lstm_cell_10/dropout/GreaterEqualЄ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_10/dropout/Cast∆
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout/Mul_1Н
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_1/Const—
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_1/MulЭ
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/ShapeР
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2оЧђ2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/yТ
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_1/GreaterEqualЊ
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_1/Castќ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_1/Mul_1Н
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_2/Const—
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_2/MulЭ
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/ShapeР
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2цЯ®2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/yТ
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_2/GreaterEqualЊ
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_2/Castќ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_2/Mul_1Н
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_3/Const—
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_3/MulЭ
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/ShapeП
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2љНN2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/yТ
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_3/GreaterEqualЊ
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_3/Castќ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_3/Mul_1К
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3§
while/lstm_cell_10/mulMulwhile_placeholder_2$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul™
while/lstm_cell_10/mul_1Mulwhile_placeholder_2&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1™
while/lstm_cell_10/mul_2Mulwhile_placeholder_2&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2™
while/lstm_cell_10/mul_3Mulwhile_placeholder_2&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
г	
І
-__inference_sequential_4_layer_call_fn_437212
input_5
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_4371762
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_5
Ќ
з
&sequential_4_lstm_10_while_cond_435491F
Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counterL
Hsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations*
&sequential_4_lstm_10_while_placeholder,
(sequential_4_lstm_10_while_placeholder_1,
(sequential_4_lstm_10_while_placeholder_2,
(sequential_4_lstm_10_while_placeholder_3H
Dsequential_4_lstm_10_while_less_sequential_4_lstm_10_strided_slice_1^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_435491___redundant_placeholder0^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_435491___redundant_placeholder1^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_435491___redundant_placeholder2^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_435491___redundant_placeholder3'
#sequential_4_lstm_10_while_identity
ў
sequential_4/lstm_10/while/LessLess&sequential_4_lstm_10_while_placeholderDsequential_4_lstm_10_while_less_sequential_4_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_4/lstm_10/while/LessЬ
#sequential_4/lstm_10/while/IdentityIdentity#sequential_4/lstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_4/lstm_10/while/Identity"S
#sequential_4_lstm_10_while_identity,sequential_4/lstm_10/while/Identity:output:0*(
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
ы≤
§	
while_body_436947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	АC
4while_lstm_cell_10_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_10_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	АA
2while_lstm_cell_10_split_1_readvariableop_resource:	А=
*while_lstm_cell_10_readvariableop_resource:	 АИҐ!while/lstm_cell_10/ReadVariableOpҐ#while/lstm_cell_10/ReadVariableOp_1Ґ#while/lstm_cell_10/ReadVariableOp_2Ґ#while/lstm_cell_10/ReadVariableOp_3Ґ'while/lstm_cell_10/split/ReadVariableOpҐ)while/lstm_cell_10/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЛ
"while/lstm_cell_10/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/ShapeН
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"while/lstm_cell_10/ones_like/Const–
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/ones_likeЙ
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2"
 while/lstm_cell_10/dropout/ConstЋ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_10/dropout/MulЩ
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/ShapeК
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2µГ°29
7while/lstm_cell_10/dropout/random_uniform/RandomUniformЫ
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2+
)while/lstm_cell_10/dropout/GreaterEqual/yК
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'while/lstm_cell_10/dropout/GreaterEqualЄ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_10/dropout/Cast∆
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout/Mul_1Н
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_1/Const—
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_1/MulЭ
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/ShapeР
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2љцЭ2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/yТ
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_1/GreaterEqualЊ
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_1/Castќ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_1/Mul_1Н
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_2/Const—
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_2/MulЭ
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/ShapeР
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2єДи2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/yТ
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_2/GreaterEqualЊ
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_2/Castќ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_2/Mul_1Н
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2$
"while/lstm_cell_10/dropout_3/Const—
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_10/dropout_3/MulЭ
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/ShapeР
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2Лѕ”2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformЯ
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/yТ
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)while/lstm_cell_10/dropout_3/GreaterEqualЊ
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_10/dropout_3/Castќ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"while/lstm_cell_10/dropout_3/Mul_1К
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim∆
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpу
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_10/split«
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMulЋ
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_1Ћ
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_2Ћ
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_3О
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim»
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpл
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1њ
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd≈
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_1≈
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_2≈
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/BiasAdd_3§
while/lstm_cell_10/mulMulwhile_placeholder_2$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul™
while/lstm_cell_10/mul_1Mulwhile_placeholder_2&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_1™
while/lstm_cell_10/mul_2Mulwhile_placeholder_2&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_2™
while/lstm_cell_10/mul_3Mulwhile_placeholder_2&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_3і
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02#
!while/lstm_cell_10/ReadVariableOp°
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack•
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1•
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2о
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceљ
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_4Ј
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/addС
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/SigmoidЄ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1•
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack©
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1©
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2ъ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1Ѕ
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_1:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_5љ
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_1Ч
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_1§
while/lstm_cell_10/mul_4Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_4Є
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2•
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack©
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1©
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2ъ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2Ѕ
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_2:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_6љ
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_2К
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Reluі
while/lstm_cell_10/mul_5Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_5Ђ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_4:z:0while/lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_3Є
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3•
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack©
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1©
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2ъ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3Ѕ
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_3:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/MatMul_7љ
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/add_4Ч
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/Relu_1Є
while/lstm_cell_10/mul_6Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_10/mul_6а
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5∆

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
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
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 
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
Ўѕ
І
C__inference_lstm_10_layer_call_and_return_conditional_losses_439113

inputs=
*lstm_cell_10_split_readvariableop_resource:	А;
,lstm_cell_10_split_1_readvariableop_resource:	А7
$lstm_cell_10_readvariableop_resource:	 А
identityИҐ=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_10/ReadVariableOpҐlstm_cell_10/ReadVariableOp_1Ґlstm_cell_10/ReadVariableOp_2Ґlstm_cell_10/ReadVariableOp_3Ґ!lstm_cell_10/split/ReadVariableOpҐ#lstm_cell_10/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2z
lstm_cell_10/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/ShapeБ
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_10/ones_like/ConstЄ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout/Const≥
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/MulЗ
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeш
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2щ®џ23
1lstm_cell_10/dropout/random_uniform/RandomUniformП
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#lstm_cell_10/dropout/GreaterEqual/yт
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_cell_10/dropout/GreaterEqual¶
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/CastЃ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout/Mul_1Б
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_1/Constє
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/MulЛ
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shapeю
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2≈ЬЫ25
3lstm_cell_10/dropout_1/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_1/GreaterEqual/yъ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_1/GreaterEqualђ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Castґ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_1/Mul_1Б
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_2/Constє
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/MulЛ
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shapeю
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2цЯЇ25
3lstm_cell_10/dropout_2/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_2/GreaterEqual/yъ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_2/GreaterEqualђ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Castґ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_2/Mul_1Б
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
lstm_cell_10/dropout_3/Constє
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/MulЛ
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shapeэ
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed“	*
seed2ЩБo25
3lstm_cell_10/dropout_3/random_uniform/RandomUniformУ
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2'
%lstm_cell_10/dropout_3/GreaterEqual/yъ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm_cell_10/dropout_3/GreaterEqualђ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Castґ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim≤
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_10/split/ReadVariableOpџ
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_10/splitЭ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul°
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_1°
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_2°
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_3В
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimі
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp”
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1І
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd≠
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_1≠
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_2≠
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/BiasAdd_3Н
lstm_cell_10/mulMulzeros:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mulУ
lstm_cell_10/mul_1Mulzeros:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_1У
lstm_cell_10/mul_2Mulzeros:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_2У
lstm_cell_10/mul_3Mulzeros:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_3†
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOpХ
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stackЩ
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1Щ
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2 
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice•
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_4Я
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid§
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_1Щ
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stackЭ
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1Э
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2÷
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1©
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_1:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_5•
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_1Е
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_1П
lstm_cell_10/mul_4Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_4§
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_2Щ
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stackЭ
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1Э
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2÷
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2©
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_2:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_6•
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_2x
lstm_cell_10/ReluRelulstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_5Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_5У
lstm_cell_10/add_3AddV2lstm_cell_10/mul_4:z:0lstm_cell_10/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_3§
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_10/ReadVariableOp_3Щ
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stackЭ
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1Э
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2÷
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3©
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_3:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/MatMul_7•
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/add_4Е
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/Relu_1†
lstm_cell_10/mul_6Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_10/mul_6П
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
while/loop_counterБ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
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
bodyR
while_body_438948*
condR
while_cond_438947*K
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
runtimeк
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	А*
dtype02?
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpџ
.lstm_10/lstm_cell_10/kernel/Regularizer/SquareSquareElstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А20
.lstm_10/lstm_cell_10/kernel/Regularizer/Squareѓ
-lstm_10/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_10/lstm_cell_10/kernel/Regularizer/Constо
+lstm_10/lstm_cell_10/kernel/Regularizer/SumSum2lstm_10/lstm_cell_10/kernel/Regularizer/Square:y:06lstm_10/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/Sum£
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82/
-lstm_10/lstm_cell_10/kernel/Regularizer/mul/xр
+lstm_10/lstm_cell_10/kernel/Regularizer/mulMul6lstm_10/lstm_cell_10/kernel/Regularizer/mul/x:output:04lstm_10/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_10/lstm_cell_10/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё
NoOpNoOp>^lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2~
=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp=lstm_10/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_436540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_436540___redundant_placeholder04
0while_while_cond_436540___redundant_placeholder14
0while_while_cond_436540___redundant_placeholder24
0while_while_cond_436540___redundant_placeholder3
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
:"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
?
input_54
serving_default_input_5:0€€€€€€€€€A
	reshape_64
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:АВ
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
regularization_losses
	variables
		keras_api


signatures
`__call__
*a&call_and_return_all_conditional_losses
b_default_save_signature"
_tf_keras_sequential
√
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
ї

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
•
trainable_variables
regularization_losses
	variables
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
'
k0"
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
 
trainable_variables

)layers
regularization_losses
*metrics
+layer_regularization_losses
	variables
,layer_metrics
-non_trainable_variables
`__call__
b_default_save_signature
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
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
0regularization_losses
1	variables
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
'
o0"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
є

3states
trainable_variables

4layers
regularization_losses
5metrics
6layer_regularization_losses
	variables
7layer_metrics
8non_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_12/kernel
: 2dense_12/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠

9layers
trainable_variables
regularization_losses
:metrics
;layer_regularization_losses
	variables
<layer_metrics
=non_trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_13/kernel
:2dense_13/bias
.
0
1"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠

>layers
trainable_variables
regularization_losses
?metrics
@layer_regularization_losses
	variables
Alayer_metrics
Bnon_trainable_variables
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
regularization_losses
Dmetrics
Elayer_regularization_losses
	variables
Flayer_metrics
Gnon_trainable_variables
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
.:,	А2lstm_10/lstm_cell_10/kernel
8:6	 А2%lstm_10/lstm_cell_10/recurrent_kernel
(:&А2lstm_10/lstm_cell_10/bias
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
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
≠

Ilayers
/trainable_variables
0regularization_losses
Jmetrics
Klayer_regularization_losses
1	variables
Llayer_metrics
Mnon_trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
k0"
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
'
o0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
&:$  2Adam/dense_12/kernel/m
 : 2Adam/dense_12/bias/m
&:$ 2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
3:1	А2"Adam/lstm_10/lstm_cell_10/kernel/m
=:;	 А2,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m
-:+А2 Adam/lstm_10/lstm_cell_10/bias/m
&:$  2Adam/dense_12/kernel/v
 : 2Adam/dense_12/bias/v
&:$ 2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
3:1	А2"Adam/lstm_10/lstm_cell_10/kernel/v
=:;	 А2,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v
-:+А2 Adam/lstm_10/lstm_cell_10/bias/v
В2€
-__inference_sequential_4_layer_call_fn_436766
-__inference_sequential_4_layer_call_fn_437338
-__inference_sequential_4_layer_call_fn_437357
-__inference_sequential_4_layer_call_fn_437212ј
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
о2л
H__inference_sequential_4_layer_call_and_return_conditional_losses_437628
H__inference_sequential_4_layer_call_and_return_conditional_losses_437963
H__inference_sequential_4_layer_call_and_return_conditional_losses_437246
H__inference_sequential_4_layer_call_and_return_conditional_losses_437280ј
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
ћB…
!__inference__wrapped_model_435641input_5"Ш
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
Г2А
(__inference_lstm_10_layer_call_fn_437980
(__inference_lstm_10_layer_call_fn_437991
(__inference_lstm_10_layer_call_fn_438002
(__inference_lstm_10_layer_call_fn_438013’
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
п2м
C__inference_lstm_10_layer_call_and_return_conditional_losses_438256
C__inference_lstm_10_layer_call_and_return_conditional_losses_438563
C__inference_lstm_10_layer_call_and_return_conditional_losses_438806
C__inference_lstm_10_layer_call_and_return_conditional_losses_439113’
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
”2–
)__inference_dense_12_layer_call_fn_439122Ґ
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
о2л
D__inference_dense_12_layer_call_and_return_conditional_losses_439133Ґ
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
”2–
)__inference_dense_13_layer_call_fn_439148Ґ
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
о2л
D__inference_dense_13_layer_call_and_return_conditional_losses_439164Ґ
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
‘2—
*__inference_reshape_6_layer_call_fn_439169Ґ
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
п2м
E__inference_reshape_6_layer_call_and_return_conditional_losses_439182Ґ
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
≥2∞
__inference_loss_fn_0_439193П
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
ЋB»
$__inference_signature_wrapper_437319input_5"Ф
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
Ґ2Я
-__inference_lstm_cell_10_layer_call_fn_439216
-__inference_lstm_cell_10_layer_call_fn_439233Њ
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
Ў2’
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439314
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439427Њ
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
≥2∞
__inference_loss_fn_1_439438П
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
annotations™ *Ґ Я
!__inference__wrapped_model_435641z&('4Ґ1
*Ґ'
%К"
input_5€€€€€€€€€
™ "9™6
4
	reshape_6'К$
	reshape_6€€€€€€€€€§
D__inference_dense_12_layer_call_and_return_conditional_losses_439133\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ |
)__inference_dense_12_layer_call_fn_439122O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ §
D__inference_dense_13_layer_call_and_return_conditional_losses_439164\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_13_layer_call_fn_439148O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€;
__inference_loss_fn_0_439193Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_439438&Ґ

Ґ 
™ "К ƒ
C__inference_lstm_10_layer_call_and_return_conditional_losses_438256}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ƒ
C__inference_lstm_10_layer_call_and_return_conditional_losses_438563}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ і
C__inference_lstm_10_layer_call_and_return_conditional_losses_438806m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ і
C__inference_lstm_10_layer_call_and_return_conditional_losses_439113m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ Ь
(__inference_lstm_10_layer_call_fn_437980p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ь
(__inference_lstm_10_layer_call_fn_437991p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ М
(__inference_lstm_10_layer_call_fn_438002`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ М
(__inference_lstm_10_layer_call_fn_438013`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€  
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439314э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ  
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_439427э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ Я
-__inference_lstm_cell_10_layer_call_fn_439216н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
1/1€€€€€€€€€ Я
-__inference_lstm_cell_10_layer_call_fn_439233н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
1/1€€€€€€€€€ •
E__inference_reshape_6_layer_call_and_return_conditional_losses_439182\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ }
*__inference_reshape_6_layer_call_fn_439169O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Њ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437246r&('<Ґ9
2Ґ/
%К"
input_5€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Њ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437280r&('<Ґ9
2Ґ/
%К"
input_5€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437628q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_4_layer_call_and_return_conditional_losses_437963q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ц
-__inference_sequential_4_layer_call_fn_436766e&('<Ґ9
2Ґ/
%К"
input_5€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ц
-__inference_sequential_4_layer_call_fn_437212e&('<Ґ9
2Ґ/
%К"
input_5€€€€€€€€€
p

 
™ "К€€€€€€€€€Х
-__inference_sequential_4_layer_call_fn_437338d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Х
-__inference_sequential_4_layer_call_fn_437357d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€Ѓ
$__inference_signature_wrapper_437319Е&('?Ґ<
Ґ 
5™2
0
input_5%К"
input_5€€€€€€€€€"9™6
4
	reshape_6'К$
	reshape_6€€€€€€€€€