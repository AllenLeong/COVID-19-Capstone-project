юы&
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8вЬ%
z
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_48/kernel
s
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes

:  *
dtype0
r
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_48/bias
k
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes
: *
dtype0
z
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_49/kernel
s
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel*
_output_shapes

: *
dtype0
r
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_49/bias
k
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
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
lstm_40/lstm_cell_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_40/lstm_cell_40/kernel

/lstm_40/lstm_cell_40/kernel/Read/ReadVariableOpReadVariableOplstm_40/lstm_cell_40/kernel*
_output_shapes
:	*
dtype0
Ї
%lstm_40/lstm_cell_40/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%lstm_40/lstm_cell_40/recurrent_kernel
 
9lstm_40/lstm_cell_40/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_40/lstm_cell_40/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_40/lstm_cell_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_40/lstm_cell_40/bias

-lstm_40/lstm_cell_40/bias/Read/ReadVariableOpReadVariableOplstm_40/lstm_cell_40/bias*
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
Adam/dense_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_48/kernel/m

*Adam/dense_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_48/bias/m
y
(Adam/dense_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/m*
_output_shapes
: *
dtype0

Adam/dense_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_49/kernel/m

*Adam/dense_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_49/bias/m
y
(Adam/dense_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/m*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_40/lstm_cell_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_40/lstm_cell_40/kernel/m

6Adam/lstm_40/lstm_cell_40/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_40/lstm_cell_40/kernel/m*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_40/lstm_cell_40/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m
Ў
@Adam/lstm_40/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

 Adam/lstm_40/lstm_cell_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_40/lstm_cell_40/bias/m

4Adam/lstm_40/lstm_cell_40/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_40/lstm_cell_40/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_48/kernel/v

*Adam/dense_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_48/bias/v
y
(Adam/dense_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/v*
_output_shapes
: *
dtype0

Adam/dense_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_49/kernel/v

*Adam/dense_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_49/bias/v
y
(Adam/dense_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/v*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_40/lstm_cell_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_40/lstm_cell_40/kernel/v

6Adam/lstm_40/lstm_cell_40/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_40/lstm_cell_40/kernel/v*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_40/lstm_cell_40/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v
Ў
@Adam/lstm_40/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

 Adam/lstm_40/lstm_cell_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_40/lstm_cell_40/bias/v

4Adam/lstm_40/lstm_cell_40/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_40/lstm_cell_40/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
е+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*+
value+B+ Bќ*
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
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
*layer_regularization_losses
	variables
trainable_variables
+layer_metrics
,metrics
regularization_losses
-non_trainable_variables
 

.
state_size

&kernel
'recurrent_kernel
(bias
/	variables
0trainable_variables
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
4layer_regularization_losses

5states
	variables
trainable_variables
6layer_metrics
7metrics
regularization_losses
8non_trainable_variables
[Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_48/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

9layers
	variables
:non_trainable_variables
trainable_variables
;layer_metrics
<metrics
regularization_losses
=layer_regularization_losses
[Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_49/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

>layers
	variables
?non_trainable_variables
trainable_variables
@layer_metrics
Ametrics
regularization_losses
Blayer_regularization_losses
 
 
 
­

Clayers
	variables
Dnon_trainable_variables
trainable_variables
Elayer_metrics
Fmetrics
regularization_losses
Glayer_regularization_losses
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
WU
VARIABLE_VALUElstm_40/lstm_cell_40/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_40/lstm_cell_40/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_40/lstm_cell_40/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 

H0
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
/	variables
Jnon_trainable_variables
0trainable_variables
Klayer_metrics
Lmetrics
1regularization_losses
Mlayer_regularization_losses
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
VARIABLE_VALUEAdam/dense_48/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_49/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_49/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_40/lstm_cell_40/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_40/lstm_cell_40/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_40/lstm_cell_40/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_48/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_49/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_49/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_40/lstm_cell_40/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_40/lstm_cell_40/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_40/lstm_cell_40/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_17Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17lstm_40/lstm_cell_40/kernellstm_40/lstm_cell_40/bias%lstm_40/lstm_cell_40/recurrent_kerneldense_48/kerneldense_48/biasdense_49/kerneldense_49/bias*
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
%__inference_signature_wrapper_1383858
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
њ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_40/lstm_cell_40/kernel/Read/ReadVariableOp9lstm_40/lstm_cell_40/recurrent_kernel/Read/ReadVariableOp-lstm_40/lstm_cell_40/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_48/kernel/m/Read/ReadVariableOp(Adam/dense_48/bias/m/Read/ReadVariableOp*Adam/dense_49/kernel/m/Read/ReadVariableOp(Adam/dense_49/bias/m/Read/ReadVariableOp6Adam/lstm_40/lstm_cell_40/kernel/m/Read/ReadVariableOp@Adam/lstm_40/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_40/lstm_cell_40/bias/m/Read/ReadVariableOp*Adam/dense_48/kernel/v/Read/ReadVariableOp(Adam/dense_48/bias/v/Read/ReadVariableOp*Adam/dense_49/kernel/v/Read/ReadVariableOp(Adam/dense_49/bias/v/Read/ReadVariableOp6Adam/lstm_40/lstm_cell_40/kernel/v/Read/ReadVariableOp@Adam/lstm_40/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_40/lstm_cell_40/bias/v/Read/ReadVariableOpConst*)
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
 __inference__traced_save_1386084
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_48/kerneldense_48/biasdense_49/kerneldense_49/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_40/lstm_cell_40/kernel%lstm_40/lstm_cell_40/recurrent_kernellstm_40/lstm_cell_40/biastotalcountAdam/dense_48/kernel/mAdam/dense_48/bias/mAdam/dense_49/kernel/mAdam/dense_49/bias/m"Adam/lstm_40/lstm_cell_40/kernel/m,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m Adam/lstm_40/lstm_cell_40/bias/mAdam/dense_48/kernel/vAdam/dense_48/bias/vAdam/dense_49/kernel/vAdam/dense_49/bias/v"Adam/lstm_40/lstm_cell_40/kernel/v,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v Adam/lstm_40/lstm_cell_40/bias/v*(
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
#__inference__traced_restore_1386178БЮ$
Ј
Ѕ	
while_body_1384662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Ѕ
while/lstm_cell_40/mulMulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЉ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Љ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Љ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
щ%
ъ
while_body_1382318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_40_1382342_0:	+
while_lstm_cell_40_1382344_0:	/
while_lstm_cell_40_1382346_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_40_1382342:	)
while_lstm_cell_40_1382344:	-
while_lstm_cell_40_1382346:	 Ђ*while/lstm_cell_40/StatefulPartitionedCallУ
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
*while/lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_40_1382342_0while_lstm_cell_40_1382344_0while_lstm_cell_40_1382346_0*
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13823042,
*while/lstm_cell_40/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_40/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_40/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_40/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_40/StatefulPartitionedCall*"
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
while_lstm_cell_40_1382342while_lstm_cell_40_1382342_0":
while_lstm_cell_40_1382344while_lstm_cell_40_1382344_0":
while_lstm_cell_40_1382346while_lstm_cell_40_1382346_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_40/StatefulPartitionedCall*while/lstm_cell_40/StatefulPartitionedCall: 
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
щ%
ъ
while_body_1382615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_40_1382639_0:	+
while_lstm_cell_40_1382641_0:	/
while_lstm_cell_40_1382643_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_40_1382639:	)
while_lstm_cell_40_1382641:	-
while_lstm_cell_40_1382643:	 Ђ*while/lstm_cell_40/StatefulPartitionedCallУ
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
*while/lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_40_1382639_0while_lstm_cell_40_1382641_0while_lstm_cell_40_1382643_0*
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13825372,
*while/lstm_cell_40/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_40/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_40/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_40/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_40/StatefulPartitionedCall*"
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
while_lstm_cell_40_1382639while_lstm_cell_40_1382639_0":
while_lstm_cell_40_1382641while_lstm_cell_40_1382641_0":
while_lstm_cell_40_1382643while_lstm_cell_40_1382643_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_40/StatefulPartitionedCall*while/lstm_cell_40/StatefulPartitionedCall: 
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
R
Щ
D__inference_lstm_40_layer_call_and_return_conditional_losses_1382393

inputs'
lstm_cell_40_1382305:	#
lstm_cell_40_1382307:	'
lstm_cell_40_1382309:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_40/StatefulPartitionedCallЂwhileD
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
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_40_1382305lstm_cell_40_1382307lstm_cell_40_1382309*
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13823042&
$lstm_cell_40/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_40_1382305lstm_cell_40_1382307lstm_cell_40_1382309*
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
while_body_1382318*
condR
while_cond_1382317*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_40_1382305*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_40/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њВ
Ѕ	
while_body_1385487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
 while/lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_40/dropout/ConstЫ
while/lstm_cell_40/dropout/MulMul%while/lstm_cell_40/ones_like:output:0)while/lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_40/dropout/Mul
 while/lstm_cell_40/dropout/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_40/dropout/Shape
7while/lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2йЃk29
7while/lstm_cell_40/dropout/random_uniform/RandomUniform
)while/lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_40/dropout/GreaterEqual/y
'while/lstm_cell_40/dropout/GreaterEqualGreaterEqual@while/lstm_cell_40/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_40/dropout/GreaterEqualИ
while/lstm_cell_40/dropout/CastCast+while/lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_40/dropout/CastЦ
 while/lstm_cell_40/dropout/Mul_1Mul"while/lstm_cell_40/dropout/Mul:z:0#while/lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout/Mul_1
"while/lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_1/Constб
 while/lstm_cell_40/dropout_1/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_1/Mul
"while/lstm_cell_40/dropout_1/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_1/Shape
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2 п2;
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_1/GreaterEqual/y
)while/lstm_cell_40/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_1/GreaterEqualО
!while/lstm_cell_40/dropout_1/CastCast-while/lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_1/CastЮ
"while/lstm_cell_40/dropout_1/Mul_1Mul$while/lstm_cell_40/dropout_1/Mul:z:0%while/lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_1/Mul_1
"while/lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_2/Constб
 while/lstm_cell_40/dropout_2/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_2/Mul
"while/lstm_cell_40/dropout_2/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_2/Shape
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ПъТ2;
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_2/GreaterEqual/y
)while/lstm_cell_40/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_2/GreaterEqualО
!while/lstm_cell_40/dropout_2/CastCast-while/lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_2/CastЮ
"while/lstm_cell_40/dropout_2/Mul_1Mul$while/lstm_cell_40/dropout_2/Mul:z:0%while/lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_2/Mul_1
"while/lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_3/Constб
 while/lstm_cell_40/dropout_3/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_3/Mul
"while/lstm_cell_40/dropout_3/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_3/Shape
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ќт2;
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_3/GreaterEqual/y
)while/lstm_cell_40/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_3/GreaterEqualО
!while/lstm_cell_40/dropout_3/CastCast-while/lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_3/CastЮ
"while/lstm_cell_40/dropout_3/Mul_1Mul$while/lstm_cell_40/dropout_3/Mul:z:0%while/lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_3/Mul_1
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Є
while/lstm_cell_40/mulMulwhile_placeholder_2$while/lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЊ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2&while/lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Њ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2&while/lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Њ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2&while/lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
ђ+
Г
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383715

inputs"
lstm_40_1383684:	
lstm_40_1383686:	"
lstm_40_1383688:	 "
dense_48_1383691:  
dense_48_1383693: "
dense_49_1383696: 
dense_49_1383698:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂlstm_40/StatefulPartitionedCallЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЅ
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinputslstm_40_1383684lstm_40_1383686lstm_40_1383688*
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13836512!
lstm_40/StatefulPartitionedCallЙ
 dense_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_48_1383691dense_48_1383693*
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
E__inference_dense_48_layer_call_and_return_conditional_losses_13832322"
 dense_48/StatefulPartitionedCallК
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1383696dense_49_1383698*
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
E__inference_dense_49_layer_call_and_return_conditional_losses_13832542"
 dense_49/StatefulPartitionedCall
reshape_24/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_13832732
reshape_24/PartitionedCallЯ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_40_1383684*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЏ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_49_1383698*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mul
IdentityIdentity#reshape_24/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall0^dense_49/bias/Regularizer/Square/ReadVariableOp ^lstm_40/StatefulPartitionedCall>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
Н
lstm_40_while_body_1384006,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3+
'lstm_40_while_lstm_40_strided_slice_1_0g
clstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0:	K
<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0:	G
4lstm_40_while_lstm_cell_40_readvariableop_resource_0:	 
lstm_40_while_identity
lstm_40_while_identity_1
lstm_40_while_identity_2
lstm_40_while_identity_3
lstm_40_while_identity_4
lstm_40_while_identity_5)
%lstm_40_while_lstm_40_strided_slice_1e
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorK
8lstm_40_while_lstm_cell_40_split_readvariableop_resource:	I
:lstm_40_while_lstm_cell_40_split_1_readvariableop_resource:	E
2lstm_40_while_lstm_cell_40_readvariableop_resource:	 Ђ)lstm_40/while/lstm_cell_40/ReadVariableOpЂ+lstm_40/while/lstm_cell_40/ReadVariableOp_1Ђ+lstm_40/while/lstm_cell_40/ReadVariableOp_2Ђ+lstm_40/while/lstm_cell_40/ReadVariableOp_3Ђ/lstm_40/while/lstm_cell_40/split/ReadVariableOpЂ1lstm_40/while/lstm_cell_40/split_1/ReadVariableOpг
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0lstm_40_while_placeholderHlstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_40/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_40/while/lstm_cell_40/ones_like/ShapeShapelstm_40_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_40/while/lstm_cell_40/ones_like/Shape
*lstm_40/while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_40/while/lstm_cell_40/ones_like/Const№
$lstm_40/while/lstm_cell_40/ones_likeFill3lstm_40/while/lstm_cell_40/ones_like/Shape:output:03lstm_40/while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/ones_like
*lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_40/while/lstm_cell_40/split/split_dimо
/lstm_40/while/lstm_cell_40/split/ReadVariableOpReadVariableOp:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_40/while/lstm_cell_40/split/ReadVariableOp
 lstm_40/while/lstm_cell_40/splitSplit3lstm_40/while/lstm_cell_40/split/split_dim:output:07lstm_40/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_40/while/lstm_cell_40/splitч
!lstm_40/while/lstm_cell_40/MatMulMatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_40/while/lstm_cell_40/MatMulы
#lstm_40/while/lstm_cell_40/MatMul_1MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_1ы
#lstm_40/while/lstm_cell_40/MatMul_2MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_2ы
#lstm_40/while/lstm_cell_40/MatMul_3MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_3
,lstm_40/while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_40/while/lstm_cell_40/split_1/split_dimр
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp
"lstm_40/while/lstm_cell_40/split_1Split5lstm_40/while/lstm_cell_40/split_1/split_dim:output:09lstm_40/while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_40/while/lstm_cell_40/split_1п
"lstm_40/while/lstm_cell_40/BiasAddBiasAdd+lstm_40/while/lstm_cell_40/MatMul:product:0+lstm_40/while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/while/lstm_cell_40/BiasAddх
$lstm_40/while/lstm_cell_40/BiasAdd_1BiasAdd-lstm_40/while/lstm_cell_40/MatMul_1:product:0+lstm_40/while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_1х
$lstm_40/while/lstm_cell_40/BiasAdd_2BiasAdd-lstm_40/while/lstm_cell_40/MatMul_2:product:0+lstm_40/while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_2х
$lstm_40/while/lstm_cell_40/BiasAdd_3BiasAdd-lstm_40/while/lstm_cell_40/MatMul_3:product:0+lstm_40/while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_3Х
lstm_40/while/lstm_cell_40/mulMullstm_40_while_placeholder_2-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/while/lstm_cell_40/mulЩ
 lstm_40/while/lstm_cell_40/mul_1Mullstm_40_while_placeholder_2-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_1Щ
 lstm_40/while/lstm_cell_40/mul_2Mullstm_40_while_placeholder_2-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_2Щ
 lstm_40/while/lstm_cell_40/mul_3Mullstm_40_while_placeholder_2-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_3Ь
)lstm_40/while/lstm_cell_40/ReadVariableOpReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_40/while/lstm_cell_40/ReadVariableOpБ
.lstm_40/while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_40/while/lstm_cell_40/strided_slice/stackЕ
0lstm_40/while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_40/while/lstm_cell_40/strided_slice/stack_1Е
0lstm_40/while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_40/while/lstm_cell_40/strided_slice/stack_2
(lstm_40/while/lstm_cell_40/strided_sliceStridedSlice1lstm_40/while/lstm_cell_40/ReadVariableOp:value:07lstm_40/while/lstm_cell_40/strided_slice/stack:output:09lstm_40/while/lstm_cell_40/strided_slice/stack_1:output:09lstm_40/while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_40/while/lstm_cell_40/strided_sliceн
#lstm_40/while/lstm_cell_40/MatMul_4MatMul"lstm_40/while/lstm_cell_40/mul:z:01lstm_40/while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_4з
lstm_40/while/lstm_cell_40/addAddV2+lstm_40/while/lstm_cell_40/BiasAdd:output:0-lstm_40/while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/while/lstm_cell_40/addЉ
"lstm_40/while/lstm_cell_40/SigmoidSigmoid"lstm_40/while/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/while/lstm_cell_40/Sigmoidа
+lstm_40/while/lstm_cell_40/ReadVariableOp_1ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_1Е
0lstm_40/while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_40/while/lstm_cell_40/strided_slice_1/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_1StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_1:value:09lstm_40/while/lstm_cell_40/strided_slice_1/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_1/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_1с
#lstm_40/while/lstm_cell_40/MatMul_5MatMul$lstm_40/while/lstm_cell_40/mul_1:z:03lstm_40/while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_5н
 lstm_40/while/lstm_cell_40/add_1AddV2-lstm_40/while/lstm_cell_40/BiasAdd_1:output:0-lstm_40/while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_1Џ
$lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid$lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/Sigmoid_1Ф
 lstm_40/while/lstm_cell_40/mul_4Mul(lstm_40/while/lstm_cell_40/Sigmoid_1:y:0lstm_40_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_4а
+lstm_40/while/lstm_cell_40/ReadVariableOp_2ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_2Е
0lstm_40/while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_40/while/lstm_cell_40/strided_slice_2/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_2StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_2:value:09lstm_40/while/lstm_cell_40/strided_slice_2/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_2/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_2с
#lstm_40/while/lstm_cell_40/MatMul_6MatMul$lstm_40/while/lstm_cell_40/mul_2:z:03lstm_40/while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_6н
 lstm_40/while/lstm_cell_40/add_2AddV2-lstm_40/while/lstm_cell_40/BiasAdd_2:output:0-lstm_40/while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_2Ђ
lstm_40/while/lstm_cell_40/ReluRelu$lstm_40/while/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_40/while/lstm_cell_40/Reluд
 lstm_40/while/lstm_cell_40/mul_5Mul&lstm_40/while/lstm_cell_40/Sigmoid:y:0-lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_5Ы
 lstm_40/while/lstm_cell_40/add_3AddV2$lstm_40/while/lstm_cell_40/mul_4:z:0$lstm_40/while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_3а
+lstm_40/while/lstm_cell_40/ReadVariableOp_3ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_3Е
0lstm_40/while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_40/while/lstm_cell_40/strided_slice_3/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_3StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_3:value:09lstm_40/while/lstm_cell_40/strided_slice_3/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_3/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_3с
#lstm_40/while/lstm_cell_40/MatMul_7MatMul$lstm_40/while/lstm_cell_40/mul_3:z:03lstm_40/while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_7н
 lstm_40/while/lstm_cell_40/add_4AddV2-lstm_40/while/lstm_cell_40/BiasAdd_3:output:0-lstm_40/while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_4Џ
$lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid$lstm_40/while/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/Sigmoid_2І
!lstm_40/while/lstm_cell_40/Relu_1Relu$lstm_40/while/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_40/while/lstm_cell_40/Relu_1и
 lstm_40/while/lstm_cell_40/mul_6Mul(lstm_40/while/lstm_cell_40/Sigmoid_2:y:0/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_6
2lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_40_while_placeholder_1lstm_40_while_placeholder$lstm_40/while/lstm_cell_40/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_40/while/TensorArrayV2Write/TensorListSetIteml
lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_40/while/add/y
lstm_40/while/addAddV2lstm_40_while_placeholderlstm_40/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_40/while/addp
lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_40/while/add_1/y
lstm_40/while/add_1AddV2(lstm_40_while_lstm_40_while_loop_counterlstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_40/while/add_1
lstm_40/while/IdentityIdentitylstm_40/while/add_1:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/IdentityІ
lstm_40/while/Identity_1Identity.lstm_40_while_lstm_40_while_maximum_iterations^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_1
lstm_40/while/Identity_2Identitylstm_40/while/add:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_2К
lstm_40/while/Identity_3IdentityBlstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_3­
lstm_40/while/Identity_4Identity$lstm_40/while/lstm_cell_40/mul_6:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/while/Identity_4­
lstm_40/while/Identity_5Identity$lstm_40/while/lstm_cell_40/add_3:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/while/Identity_5
lstm_40/while/NoOpNoOp*^lstm_40/while/lstm_cell_40/ReadVariableOp,^lstm_40/while/lstm_cell_40/ReadVariableOp_1,^lstm_40/while/lstm_cell_40/ReadVariableOp_2,^lstm_40/while/lstm_cell_40/ReadVariableOp_30^lstm_40/while/lstm_cell_40/split/ReadVariableOp2^lstm_40/while/lstm_cell_40/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_40/while/NoOp"9
lstm_40_while_identitylstm_40/while/Identity:output:0"=
lstm_40_while_identity_1!lstm_40/while/Identity_1:output:0"=
lstm_40_while_identity_2!lstm_40/while/Identity_2:output:0"=
lstm_40_while_identity_3!lstm_40/while/Identity_3:output:0"=
lstm_40_while_identity_4!lstm_40/while/Identity_4:output:0"=
lstm_40_while_identity_5!lstm_40/while/Identity_5:output:0"P
%lstm_40_while_lstm_40_strided_slice_1'lstm_40_while_lstm_40_strided_slice_1_0"j
2lstm_40_while_lstm_cell_40_readvariableop_resource4lstm_40_while_lstm_cell_40_readvariableop_resource_0"z
:lstm_40_while_lstm_cell_40_split_1_readvariableop_resource<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0"v
8lstm_40_while_lstm_cell_40_split_readvariableop_resource:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0"Ш
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_40/while/lstm_cell_40/ReadVariableOp)lstm_40/while/lstm_cell_40/ReadVariableOp2Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_1+lstm_40/while/lstm_cell_40/ReadVariableOp_12Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_2+lstm_40/while/lstm_cell_40/ReadVariableOp_22Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_3+lstm_40/while/lstm_cell_40/ReadVariableOp_32b
/lstm_40/while/lstm_cell_40/split/ReadVariableOp/lstm_40/while/lstm_cell_40/split/ReadVariableOp2f
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp: 
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
Иv
ь
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385932

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
seed2Яйѕ2&
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
seed2Г2(
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
seed2ЇЎ2(
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
seed2ЋУ2(
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2,
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
ј+
Е
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383785
input_17"
lstm_40_1383754:	
lstm_40_1383756:	"
lstm_40_1383758:	 "
dense_48_1383761:  
dense_48_1383763: "
dense_49_1383766: 
dense_49_1383768:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂlstm_40/StatefulPartitionedCallЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЇ
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinput_17lstm_40_1383754lstm_40_1383756lstm_40_1383758*
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13832132!
lstm_40/StatefulPartitionedCallЙ
 dense_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_48_1383761dense_48_1383763*
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
E__inference_dense_48_layer_call_and_return_conditional_losses_13832322"
 dense_48/StatefulPartitionedCallК
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1383766dense_49_1383768*
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
E__inference_dense_49_layer_call_and_return_conditional_losses_13832542"
 dense_49/StatefulPartitionedCall
reshape_24/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_13832732
reshape_24/PartitionedCallЯ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_40_1383754*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЏ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_49_1383768*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mul
IdentityIdentity#reshape_24/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall0^dense_49/bias/Regularizer/Square/ReadVariableOp ^lstm_40/StatefulPartitionedCall>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_17
№

(sequential_16_lstm_40_while_cond_1382030H
Dsequential_16_lstm_40_while_sequential_16_lstm_40_while_loop_counterN
Jsequential_16_lstm_40_while_sequential_16_lstm_40_while_maximum_iterations+
'sequential_16_lstm_40_while_placeholder-
)sequential_16_lstm_40_while_placeholder_1-
)sequential_16_lstm_40_while_placeholder_2-
)sequential_16_lstm_40_while_placeholder_3J
Fsequential_16_lstm_40_while_less_sequential_16_lstm_40_strided_slice_1a
]sequential_16_lstm_40_while_sequential_16_lstm_40_while_cond_1382030___redundant_placeholder0a
]sequential_16_lstm_40_while_sequential_16_lstm_40_while_cond_1382030___redundant_placeholder1a
]sequential_16_lstm_40_while_sequential_16_lstm_40_while_cond_1382030___redundant_placeholder2a
]sequential_16_lstm_40_while_sequential_16_lstm_40_while_cond_1382030___redundant_placeholder3(
$sequential_16_lstm_40_while_identity
о
 sequential_16/lstm_40/while/LessLess'sequential_16_lstm_40_while_placeholderFsequential_16_lstm_40_while_less_sequential_16_lstm_40_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_16/lstm_40/while/Less
$sequential_16/lstm_40/while/IdentityIdentity$sequential_16/lstm_40/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_16/lstm_40/while/Identity"U
$sequential_16_lstm_40_while_identity-sequential_16/lstm_40/while/Identity:output:0*(
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
)__inference_lstm_40_layer_call_fn_1384552

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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13836512
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

і
E__inference_dense_48_layer_call_and_return_conditional_losses_1383232

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
ћВ
Ѕ	
while_body_1384937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
 while/lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_40/dropout/ConstЫ
while/lstm_cell_40/dropout/MulMul%while/lstm_cell_40/ones_like:output:0)while/lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_40/dropout/Mul
 while/lstm_cell_40/dropout/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_40/dropout/Shape
7while/lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Њ29
7while/lstm_cell_40/dropout/random_uniform/RandomUniform
)while/lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_40/dropout/GreaterEqual/y
'while/lstm_cell_40/dropout/GreaterEqualGreaterEqual@while/lstm_cell_40/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_40/dropout/GreaterEqualИ
while/lstm_cell_40/dropout/CastCast+while/lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_40/dropout/CastЦ
 while/lstm_cell_40/dropout/Mul_1Mul"while/lstm_cell_40/dropout/Mul:z:0#while/lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout/Mul_1
"while/lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_1/Constб
 while/lstm_cell_40/dropout_1/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_1/Mul
"while/lstm_cell_40/dropout_1/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_1/Shape
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ос2;
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_1/GreaterEqual/y
)while/lstm_cell_40/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_1/GreaterEqualО
!while/lstm_cell_40/dropout_1/CastCast-while/lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_1/CastЮ
"while/lstm_cell_40/dropout_1/Mul_1Mul$while/lstm_cell_40/dropout_1/Mul:z:0%while/lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_1/Mul_1
"while/lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_2/Constб
 while/lstm_cell_40/dropout_2/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_2/Mul
"while/lstm_cell_40/dropout_2/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_2/Shape
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ё2;
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_2/GreaterEqual/y
)while/lstm_cell_40/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_2/GreaterEqualО
!while/lstm_cell_40/dropout_2/CastCast-while/lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_2/CastЮ
"while/lstm_cell_40/dropout_2/Mul_1Mul$while/lstm_cell_40/dropout_2/Mul:z:0%while/lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_2/Mul_1
"while/lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_3/Constб
 while/lstm_cell_40/dropout_3/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_3/Mul
"while/lstm_cell_40/dropout_3/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_3/Shape
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2чы/2;
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_3/GreaterEqual/y
)while/lstm_cell_40/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_3/GreaterEqualО
!while/lstm_cell_40/dropout_3/CastCast-while/lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_3/CastЮ
"while/lstm_cell_40/dropout_3/Mul_1Mul$while/lstm_cell_40/dropout_3/Mul:z:0%while/lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_3/Mul_1
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Є
while/lstm_cell_40/mulMulwhile_placeholder_2$while/lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЊ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2&while/lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Њ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2&while/lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Њ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2&while/lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
Р
И
)__inference_lstm_40_layer_call_fn_1384530
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13826902
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
сЁ
Ј
D__inference_lstm_40_layer_call_and_return_conditional_losses_1383213

inputs=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileD
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1383080*
condR
while_cond_1383079*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_1383485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1383485___redundant_placeholder05
1while_while_cond_1383485___redundant_placeholder15
1while_while_cond_1383485___redundant_placeholder25
1while_while_cond_1383485___redundant_placeholder3
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


J__inference_sequential_16_layer_call_and_return_conditional_losses_1384502

inputsE
2lstm_40_lstm_cell_40_split_readvariableop_resource:	C
4lstm_40_lstm_cell_40_split_1_readvariableop_resource:	?
,lstm_40_lstm_cell_40_readvariableop_resource:	 9
'dense_48_matmul_readvariableop_resource:  6
(dense_48_biasadd_readvariableop_resource: 9
'dense_49_matmul_readvariableop_resource: 6
(dense_49_biasadd_readvariableop_resource:
identityЂdense_48/BiasAdd/ReadVariableOpЂdense_48/MatMul/ReadVariableOpЂdense_49/BiasAdd/ReadVariableOpЂdense_49/MatMul/ReadVariableOpЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂ#lstm_40/lstm_cell_40/ReadVariableOpЂ%lstm_40/lstm_cell_40/ReadVariableOp_1Ђ%lstm_40/lstm_cell_40/ReadVariableOp_2Ђ%lstm_40/lstm_cell_40/ReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_40/lstm_cell_40/split/ReadVariableOpЂ+lstm_40/lstm_cell_40/split_1/ReadVariableOpЂlstm_40/whileT
lstm_40/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_40/Shape
lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice/stack
lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_40/strided_slice/stack_1
lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_40/strided_slice/stack_2
lstm_40/strided_sliceStridedSlicelstm_40/Shape:output:0$lstm_40/strided_slice/stack:output:0&lstm_40/strided_slice/stack_1:output:0&lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_40/strided_slicel
lstm_40/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros/mul/y
lstm_40/zeros/mulMullstm_40/strided_slice:output:0lstm_40/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros/mulo
lstm_40/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_40/zeros/Less/y
lstm_40/zeros/LessLesslstm_40/zeros/mul:z:0lstm_40/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros/Lessr
lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros/packed/1Ѓ
lstm_40/zeros/packedPacklstm_40/strided_slice:output:0lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_40/zeros/packedo
lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/zeros/Const
lstm_40/zerosFilllstm_40/zeros/packed:output:0lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/zerosp
lstm_40/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros_1/mul/y
lstm_40/zeros_1/mulMullstm_40/strided_slice:output:0lstm_40/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros_1/muls
lstm_40/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_40/zeros_1/Less/y
lstm_40/zeros_1/LessLesslstm_40/zeros_1/mul:z:0lstm_40/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros_1/Lessv
lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros_1/packed/1Љ
lstm_40/zeros_1/packedPacklstm_40/strided_slice:output:0!lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_40/zeros_1/packeds
lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/zeros_1/Const
lstm_40/zeros_1Filllstm_40/zeros_1/packed:output:0lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/zeros_1
lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_40/transpose/perm
lstm_40/transpose	Transposeinputslstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_40/transposeg
lstm_40/Shape_1Shapelstm_40/transpose:y:0*
T0*
_output_shapes
:2
lstm_40/Shape_1
lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice_1/stack
lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_1/stack_1
lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_1/stack_2
lstm_40/strided_slice_1StridedSlicelstm_40/Shape_1:output:0&lstm_40/strided_slice_1/stack:output:0(lstm_40/strided_slice_1/stack_1:output:0(lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_40/strided_slice_1
#lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_40/TensorArrayV2/element_shapeв
lstm_40/TensorArrayV2TensorListReserve,lstm_40/TensorArrayV2/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_40/TensorArrayV2Я
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_40/transpose:y:0Flstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_40/TensorArrayUnstack/TensorListFromTensor
lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice_2/stack
lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_2/stack_1
lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_2/stack_2Ќ
lstm_40/strided_slice_2StridedSlicelstm_40/transpose:y:0&lstm_40/strided_slice_2/stack:output:0(lstm_40/strided_slice_2/stack_1:output:0(lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_40/strided_slice_2
$lstm_40/lstm_cell_40/ones_like/ShapeShapelstm_40/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_40/lstm_cell_40/ones_like/Shape
$lstm_40/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_40/lstm_cell_40/ones_like/Constи
lstm_40/lstm_cell_40/ones_likeFill-lstm_40/lstm_cell_40/ones_like/Shape:output:0-lstm_40/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/ones_like
"lstm_40/lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"lstm_40/lstm_cell_40/dropout/Constг
 lstm_40/lstm_cell_40/dropout/MulMul'lstm_40/lstm_cell_40/ones_like:output:0+lstm_40/lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/lstm_cell_40/dropout/Mul
"lstm_40/lstm_cell_40/dropout/ShapeShape'lstm_40/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_40/lstm_cell_40/dropout/Shape
9lstm_40/lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform+lstm_40/lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2є	2;
9lstm_40/lstm_cell_40/dropout/random_uniform/RandomUniform
+lstm_40/lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+lstm_40/lstm_cell_40/dropout/GreaterEqual/y
)lstm_40/lstm_cell_40/dropout/GreaterEqualGreaterEqualBlstm_40/lstm_cell_40/dropout/random_uniform/RandomUniform:output:04lstm_40/lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_40/lstm_cell_40/dropout/GreaterEqualО
!lstm_40/lstm_cell_40/dropout/CastCast-lstm_40/lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_40/lstm_cell_40/dropout/CastЮ
"lstm_40/lstm_cell_40/dropout/Mul_1Mul$lstm_40/lstm_cell_40/dropout/Mul:z:0%lstm_40/lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/lstm_cell_40/dropout/Mul_1
$lstm_40/lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_40/lstm_cell_40/dropout_1/Constй
"lstm_40/lstm_cell_40/dropout_1/MulMul'lstm_40/lstm_cell_40/ones_like:output:0-lstm_40/lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/lstm_cell_40/dropout_1/MulЃ
$lstm_40/lstm_cell_40/dropout_1/ShapeShape'lstm_40/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_40/lstm_cell_40/dropout_1/Shape
;lstm_40/lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_40/lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Шш\2=
;lstm_40/lstm_cell_40/dropout_1/random_uniform/RandomUniformЃ
-lstm_40/lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_40/lstm_cell_40/dropout_1/GreaterEqual/y
+lstm_40/lstm_cell_40/dropout_1/GreaterEqualGreaterEqualDlstm_40/lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:06lstm_40/lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_40/lstm_cell_40/dropout_1/GreaterEqualФ
#lstm_40/lstm_cell_40/dropout_1/CastCast/lstm_40/lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/lstm_cell_40/dropout_1/Castж
$lstm_40/lstm_cell_40/dropout_1/Mul_1Mul&lstm_40/lstm_cell_40/dropout_1/Mul:z:0'lstm_40/lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/lstm_cell_40/dropout_1/Mul_1
$lstm_40/lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_40/lstm_cell_40/dropout_2/Constй
"lstm_40/lstm_cell_40/dropout_2/MulMul'lstm_40/lstm_cell_40/ones_like:output:0-lstm_40/lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/lstm_cell_40/dropout_2/MulЃ
$lstm_40/lstm_cell_40/dropout_2/ShapeShape'lstm_40/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_40/lstm_cell_40/dropout_2/Shape
;lstm_40/lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_40/lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ьѓт2=
;lstm_40/lstm_cell_40/dropout_2/random_uniform/RandomUniformЃ
-lstm_40/lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_40/lstm_cell_40/dropout_2/GreaterEqual/y
+lstm_40/lstm_cell_40/dropout_2/GreaterEqualGreaterEqualDlstm_40/lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:06lstm_40/lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_40/lstm_cell_40/dropout_2/GreaterEqualФ
#lstm_40/lstm_cell_40/dropout_2/CastCast/lstm_40/lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/lstm_cell_40/dropout_2/Castж
$lstm_40/lstm_cell_40/dropout_2/Mul_1Mul&lstm_40/lstm_cell_40/dropout_2/Mul:z:0'lstm_40/lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/lstm_cell_40/dropout_2/Mul_1
$lstm_40/lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_40/lstm_cell_40/dropout_3/Constй
"lstm_40/lstm_cell_40/dropout_3/MulMul'lstm_40/lstm_cell_40/ones_like:output:0-lstm_40/lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/lstm_cell_40/dropout_3/MulЃ
$lstm_40/lstm_cell_40/dropout_3/ShapeShape'lstm_40/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_40/lstm_cell_40/dropout_3/Shape
;lstm_40/lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_40/lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2мЮ2=
;lstm_40/lstm_cell_40/dropout_3/random_uniform/RandomUniformЃ
-lstm_40/lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_40/lstm_cell_40/dropout_3/GreaterEqual/y
+lstm_40/lstm_cell_40/dropout_3/GreaterEqualGreaterEqualDlstm_40/lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:06lstm_40/lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_40/lstm_cell_40/dropout_3/GreaterEqualФ
#lstm_40/lstm_cell_40/dropout_3/CastCast/lstm_40/lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/lstm_cell_40/dropout_3/Castж
$lstm_40/lstm_cell_40/dropout_3/Mul_1Mul&lstm_40/lstm_cell_40/dropout_3/Mul:z:0'lstm_40/lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/lstm_cell_40/dropout_3/Mul_1
$lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_40/lstm_cell_40/split/split_dimЪ
)lstm_40/lstm_cell_40/split/ReadVariableOpReadVariableOp2lstm_40_lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_40/lstm_cell_40/split/ReadVariableOpћ
lstm_40/lstm_cell_40/splitSplit-lstm_40/lstm_cell_40/split/split_dim:output:01lstm_40/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_40/lstm_cell_40/splitН
lstm_40/lstm_cell_40/MatMulMatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMulС
lstm_40/lstm_cell_40/MatMul_1MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_1С
lstm_40/lstm_cell_40/MatMul_2MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_2С
lstm_40/lstm_cell_40/MatMul_3MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_3
&lstm_40/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_40/lstm_cell_40/split_1/split_dimЬ
+lstm_40/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4lstm_40_lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_40/lstm_cell_40/split_1/ReadVariableOpѓ
lstm_40/lstm_cell_40/split_1Split/lstm_40/lstm_cell_40/split_1/split_dim:output:03lstm_40/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_40/lstm_cell_40/split_1Ч
lstm_40/lstm_cell_40/BiasAddBiasAdd%lstm_40/lstm_cell_40/MatMul:product:0%lstm_40/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/BiasAddЭ
lstm_40/lstm_cell_40/BiasAdd_1BiasAdd'lstm_40/lstm_cell_40/MatMul_1:product:0%lstm_40/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_1Э
lstm_40/lstm_cell_40/BiasAdd_2BiasAdd'lstm_40/lstm_cell_40/MatMul_2:product:0%lstm_40/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_2Э
lstm_40/lstm_cell_40/BiasAdd_3BiasAdd'lstm_40/lstm_cell_40/MatMul_3:product:0%lstm_40/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_3­
lstm_40/lstm_cell_40/mulMullstm_40/zeros:output:0&lstm_40/lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mulГ
lstm_40/lstm_cell_40/mul_1Mullstm_40/zeros:output:0(lstm_40/lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_1Г
lstm_40/lstm_cell_40/mul_2Mullstm_40/zeros:output:0(lstm_40/lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_2Г
lstm_40/lstm_cell_40/mul_3Mullstm_40/zeros:output:0(lstm_40/lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_3И
#lstm_40/lstm_cell_40/ReadVariableOpReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_40/lstm_cell_40/ReadVariableOpЅ
(lstm_40/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_40/lstm_cell_40/strided_slice/stackЉ
*lstm_40/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_40/lstm_cell_40/strided_slice/stack_1Љ
*lstm_40/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_40/lstm_cell_40/strided_slice/stack_2њ
"lstm_40/lstm_cell_40/strided_sliceStridedSlice+lstm_40/lstm_cell_40/ReadVariableOp:value:01lstm_40/lstm_cell_40/strided_slice/stack:output:03lstm_40/lstm_cell_40/strided_slice/stack_1:output:03lstm_40/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_40/lstm_cell_40/strided_sliceХ
lstm_40/lstm_cell_40/MatMul_4MatMullstm_40/lstm_cell_40/mul:z:0+lstm_40/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_4П
lstm_40/lstm_cell_40/addAddV2%lstm_40/lstm_cell_40/BiasAdd:output:0'lstm_40/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add
lstm_40/lstm_cell_40/SigmoidSigmoidlstm_40/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/SigmoidМ
%lstm_40/lstm_cell_40/ReadVariableOp_1ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_1Љ
*lstm_40/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_40/lstm_cell_40/strided_slice_1/stack­
,lstm_40/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_40/lstm_cell_40/strided_slice_1/stack_1­
,lstm_40/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_1/stack_2
$lstm_40/lstm_cell_40/strided_slice_1StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_1:value:03lstm_40/lstm_cell_40/strided_slice_1/stack:output:05lstm_40/lstm_cell_40/strided_slice_1/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_1Щ
lstm_40/lstm_cell_40/MatMul_5MatMullstm_40/lstm_cell_40/mul_1:z:0-lstm_40/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_5Х
lstm_40/lstm_cell_40/add_1AddV2'lstm_40/lstm_cell_40/BiasAdd_1:output:0'lstm_40/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_1
lstm_40/lstm_cell_40/Sigmoid_1Sigmoidlstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/Sigmoid_1Џ
lstm_40/lstm_cell_40/mul_4Mul"lstm_40/lstm_cell_40/Sigmoid_1:y:0lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_4М
%lstm_40/lstm_cell_40/ReadVariableOp_2ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_2Љ
*lstm_40/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_40/lstm_cell_40/strided_slice_2/stack­
,lstm_40/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_40/lstm_cell_40/strided_slice_2/stack_1­
,lstm_40/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_2/stack_2
$lstm_40/lstm_cell_40/strided_slice_2StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_2:value:03lstm_40/lstm_cell_40/strided_slice_2/stack:output:05lstm_40/lstm_cell_40/strided_slice_2/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_2Щ
lstm_40/lstm_cell_40/MatMul_6MatMullstm_40/lstm_cell_40/mul_2:z:0-lstm_40/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_6Х
lstm_40/lstm_cell_40/add_2AddV2'lstm_40/lstm_cell_40/BiasAdd_2:output:0'lstm_40/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_2
lstm_40/lstm_cell_40/ReluRelulstm_40/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/ReluМ
lstm_40/lstm_cell_40/mul_5Mul lstm_40/lstm_cell_40/Sigmoid:y:0'lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_5Г
lstm_40/lstm_cell_40/add_3AddV2lstm_40/lstm_cell_40/mul_4:z:0lstm_40/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_3М
%lstm_40/lstm_cell_40/ReadVariableOp_3ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_3Љ
*lstm_40/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_40/lstm_cell_40/strided_slice_3/stack­
,lstm_40/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_40/lstm_cell_40/strided_slice_3/stack_1­
,lstm_40/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_3/stack_2
$lstm_40/lstm_cell_40/strided_slice_3StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_3:value:03lstm_40/lstm_cell_40/strided_slice_3/stack:output:05lstm_40/lstm_cell_40/strided_slice_3/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_3Щ
lstm_40/lstm_cell_40/MatMul_7MatMullstm_40/lstm_cell_40/mul_3:z:0-lstm_40/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_7Х
lstm_40/lstm_cell_40/add_4AddV2'lstm_40/lstm_cell_40/BiasAdd_3:output:0'lstm_40/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_4
lstm_40/lstm_cell_40/Sigmoid_2Sigmoidlstm_40/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/Sigmoid_2
lstm_40/lstm_cell_40/Relu_1Relulstm_40/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/Relu_1Р
lstm_40/lstm_cell_40/mul_6Mul"lstm_40/lstm_cell_40/Sigmoid_2:y:0)lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_6
%lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_40/TensorArrayV2_1/element_shapeи
lstm_40/TensorArrayV2_1TensorListReserve.lstm_40/TensorArrayV2_1/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_40/TensorArrayV2_1^
lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/time
 lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_40/while/maximum_iterationsz
lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/while/loop_counterћ
lstm_40/whileWhile#lstm_40/while/loop_counter:output:0)lstm_40/while/maximum_iterations:output:0lstm_40/time:output:0 lstm_40/TensorArrayV2_1:handle:0lstm_40/zeros:output:0lstm_40/zeros_1:output:0 lstm_40/strided_slice_1:output:0?lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_40_lstm_cell_40_split_readvariableop_resource4lstm_40_lstm_cell_40_split_1_readvariableop_resource,lstm_40_lstm_cell_40_readvariableop_resource*
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
lstm_40_while_body_1384309*&
condR
lstm_40_while_cond_1384308*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_40/whileХ
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_40/TensorArrayV2Stack/TensorListStackTensorListStacklstm_40/while:output:3Alstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_40/TensorArrayV2Stack/TensorListStack
lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_40/strided_slice_3/stack
lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_40/strided_slice_3/stack_1
lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_3/stack_2Ъ
lstm_40/strided_slice_3StridedSlice3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_40/strided_slice_3/stack:output:0(lstm_40/strided_slice_3/stack_1:output:0(lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_40/strided_slice_3
lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_40/transpose_1/permХ
lstm_40/transpose_1	Transpose3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_40/transpose_1v
lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/runtimeЈ
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_48/MatMul/ReadVariableOpЈ
dense_48/MatMulMatMul lstm_40/strided_slice_3:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/MatMulЇ
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_48/BiasAdd/ReadVariableOpЅ
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/BiasAdds
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/ReluЈ
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_49/MatMul/ReadVariableOpЃ
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_49/MatMulЇ
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOpЅ
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_49/BiasAddm
reshape_24/ShapeShapedense_49/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_24/Shape
reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_24/strided_slice/stack
 reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_1
 reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_2Є
reshape_24/strided_sliceStridedSlicereshape_24/Shape:output:0'reshape_24/strided_slice/stack:output:0)reshape_24/strided_slice/stack_1:output:0)reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_24/strided_slicez
reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/1z
reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/2з
reshape_24/Reshape/shapePack!reshape_24/strided_slice:output:0#reshape_24/Reshape/shape/1:output:0#reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_24/Reshape/shapeЇ
reshape_24/ReshapeReshapedense_49/BiasAdd:output:0!reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_24/Reshapeђ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_40_lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЧ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mulz
IdentityIdentityreshape_24/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp0^dense_49/bias/Regularizer/Square/ReadVariableOp$^lstm_40/lstm_cell_40/ReadVariableOp&^lstm_40/lstm_cell_40/ReadVariableOp_1&^lstm_40/lstm_cell_40/ReadVariableOp_2&^lstm_40/lstm_cell_40/ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*^lstm_40/lstm_cell_40/split/ReadVariableOp,^lstm_40/lstm_cell_40/split_1/ReadVariableOp^lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2J
#lstm_40/lstm_cell_40/ReadVariableOp#lstm_40/lstm_cell_40/ReadVariableOp2N
%lstm_40/lstm_cell_40/ReadVariableOp_1%lstm_40/lstm_cell_40/ReadVariableOp_12N
%lstm_40/lstm_cell_40/ReadVariableOp_2%lstm_40/lstm_cell_40/ReadVariableOp_22N
%lstm_40/lstm_cell_40/ReadVariableOp_3%lstm_40/lstm_cell_40/ReadVariableOp_32~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_40/lstm_cell_40/split/ReadVariableOp)lstm_40/lstm_cell_40/split/ReadVariableOp2Z
+lstm_40/lstm_cell_40/split_1/ReadVariableOp+lstm_40/lstm_cell_40/split_1/ReadVariableOp2
lstm_40/whilelstm_40/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И	
 
%__inference_signature_wrapper_1383858
input_17
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
"__inference__wrapped_model_13821802
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
input_17
Ш{

#__inference__traced_restore_1386178
file_prefix2
 assignvariableop_dense_48_kernel:  .
 assignvariableop_1_dense_48_bias: 4
"assignvariableop_2_dense_49_kernel: .
 assignvariableop_3_dense_49_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_40_lstm_cell_40_kernel:	L
9assignvariableop_10_lstm_40_lstm_cell_40_recurrent_kernel:	 <
-assignvariableop_11_lstm_40_lstm_cell_40_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_48_kernel_m:  6
(assignvariableop_15_adam_dense_48_bias_m: <
*assignvariableop_16_adam_dense_49_kernel_m: 6
(assignvariableop_17_adam_dense_49_bias_m:I
6assignvariableop_18_adam_lstm_40_lstm_cell_40_kernel_m:	S
@assignvariableop_19_adam_lstm_40_lstm_cell_40_recurrent_kernel_m:	 C
4assignvariableop_20_adam_lstm_40_lstm_cell_40_bias_m:	<
*assignvariableop_21_adam_dense_48_kernel_v:  6
(assignvariableop_22_adam_dense_48_bias_v: <
*assignvariableop_23_adam_dense_49_kernel_v: 6
(assignvariableop_24_adam_dense_49_bias_v:I
6assignvariableop_25_adam_lstm_40_lstm_cell_40_kernel_v:	S
@assignvariableop_26_adam_lstm_40_lstm_cell_40_recurrent_kernel_v:	 C
4assignvariableop_27_adam_lstm_40_lstm_cell_40_bias_v:	
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueўBћB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_48_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_48_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_49_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_49_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_40_lstm_cell_40_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10С
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_40_lstm_cell_40_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_40_lstm_cell_40_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_48_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15А
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_48_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_49_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_49_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18О
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_40_lstm_cell_40_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ш
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_40_lstm_cell_40_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20М
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_40_lstm_cell_40_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_48_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_48_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_49_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_49_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25О
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_40_lstm_cell_40_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ш
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_40_lstm_cell_40_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27М
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_40_lstm_cell_40_bias_vIdentity_27:output:0"/device:CPU:0*
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
Ј
Ѕ	
while_body_1385212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Ѕ
while/lstm_cell_40/mulMulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЉ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Љ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Љ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
ёЭ
Н
lstm_40_while_body_1384309,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3+
'lstm_40_while_lstm_40_strided_slice_1_0g
clstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0:	K
<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0:	G
4lstm_40_while_lstm_cell_40_readvariableop_resource_0:	 
lstm_40_while_identity
lstm_40_while_identity_1
lstm_40_while_identity_2
lstm_40_while_identity_3
lstm_40_while_identity_4
lstm_40_while_identity_5)
%lstm_40_while_lstm_40_strided_slice_1e
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorK
8lstm_40_while_lstm_cell_40_split_readvariableop_resource:	I
:lstm_40_while_lstm_cell_40_split_1_readvariableop_resource:	E
2lstm_40_while_lstm_cell_40_readvariableop_resource:	 Ђ)lstm_40/while/lstm_cell_40/ReadVariableOpЂ+lstm_40/while/lstm_cell_40/ReadVariableOp_1Ђ+lstm_40/while/lstm_cell_40/ReadVariableOp_2Ђ+lstm_40/while/lstm_cell_40/ReadVariableOp_3Ђ/lstm_40/while/lstm_cell_40/split/ReadVariableOpЂ1lstm_40/while/lstm_cell_40/split_1/ReadVariableOpг
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0lstm_40_while_placeholderHlstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_40/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_40/while/lstm_cell_40/ones_like/ShapeShapelstm_40_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_40/while/lstm_cell_40/ones_like/Shape
*lstm_40/while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_40/while/lstm_cell_40/ones_like/Const№
$lstm_40/while/lstm_cell_40/ones_likeFill3lstm_40/while/lstm_cell_40/ones_like/Shape:output:03lstm_40/while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/ones_like
(lstm_40/while/lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(lstm_40/while/lstm_cell_40/dropout/Constы
&lstm_40/while/lstm_cell_40/dropout/MulMul-lstm_40/while/lstm_cell_40/ones_like:output:01lstm_40/while/lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_40/while/lstm_cell_40/dropout/MulБ
(lstm_40/while/lstm_cell_40/dropout/ShapeShape-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_40/while/lstm_cell_40/dropout/ShapeЂ
?lstm_40/while/lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform1lstm_40/while/lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Њ2A
?lstm_40/while/lstm_cell_40/dropout/random_uniform/RandomUniformЋ
1lstm_40/while/lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1lstm_40/while/lstm_cell_40/dropout/GreaterEqual/yЊ
/lstm_40/while/lstm_cell_40/dropout/GreaterEqualGreaterEqualHlstm_40/while/lstm_cell_40/dropout/random_uniform/RandomUniform:output:0:lstm_40/while/lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_40/while/lstm_cell_40/dropout/GreaterEqualа
'lstm_40/while/lstm_cell_40/dropout/CastCast3lstm_40/while/lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_40/while/lstm_cell_40/dropout/Castц
(lstm_40/while/lstm_cell_40/dropout/Mul_1Mul*lstm_40/while/lstm_cell_40/dropout/Mul:z:0+lstm_40/while/lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_40/while/lstm_cell_40/dropout/Mul_1
*lstm_40/while/lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_40/while/lstm_cell_40/dropout_1/Constё
(lstm_40/while/lstm_cell_40/dropout_1/MulMul-lstm_40/while/lstm_cell_40/ones_like:output:03lstm_40/while/lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_40/while/lstm_cell_40/dropout_1/MulЕ
*lstm_40/while/lstm_cell_40/dropout_1/ShapeShape-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_40/while/lstm_cell_40/dropout_1/ShapeЈ
Alstm_40/while/lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_40/while/lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Хђ2C
Alstm_40/while/lstm_cell_40/dropout_1/random_uniform/RandomUniformЏ
3lstm_40/while/lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_40/while/lstm_cell_40/dropout_1/GreaterEqual/yВ
1lstm_40/while/lstm_cell_40/dropout_1/GreaterEqualGreaterEqualJlstm_40/while/lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:0<lstm_40/while/lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_40/while/lstm_cell_40/dropout_1/GreaterEqualж
)lstm_40/while/lstm_cell_40/dropout_1/CastCast5lstm_40/while/lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_40/while/lstm_cell_40/dropout_1/Castю
*lstm_40/while/lstm_cell_40/dropout_1/Mul_1Mul,lstm_40/while/lstm_cell_40/dropout_1/Mul:z:0-lstm_40/while/lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_40/while/lstm_cell_40/dropout_1/Mul_1
*lstm_40/while/lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_40/while/lstm_cell_40/dropout_2/Constё
(lstm_40/while/lstm_cell_40/dropout_2/MulMul-lstm_40/while/lstm_cell_40/ones_like:output:03lstm_40/while/lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_40/while/lstm_cell_40/dropout_2/MulЕ
*lstm_40/while/lstm_cell_40/dropout_2/ShapeShape-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_40/while/lstm_cell_40/dropout_2/ShapeЈ
Alstm_40/while/lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_40/while/lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2яо2C
Alstm_40/while/lstm_cell_40/dropout_2/random_uniform/RandomUniformЏ
3lstm_40/while/lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_40/while/lstm_cell_40/dropout_2/GreaterEqual/yВ
1lstm_40/while/lstm_cell_40/dropout_2/GreaterEqualGreaterEqualJlstm_40/while/lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:0<lstm_40/while/lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_40/while/lstm_cell_40/dropout_2/GreaterEqualж
)lstm_40/while/lstm_cell_40/dropout_2/CastCast5lstm_40/while/lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_40/while/lstm_cell_40/dropout_2/Castю
*lstm_40/while/lstm_cell_40/dropout_2/Mul_1Mul,lstm_40/while/lstm_cell_40/dropout_2/Mul:z:0-lstm_40/while/lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_40/while/lstm_cell_40/dropout_2/Mul_1
*lstm_40/while/lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_40/while/lstm_cell_40/dropout_3/Constё
(lstm_40/while/lstm_cell_40/dropout_3/MulMul-lstm_40/while/lstm_cell_40/ones_like:output:03lstm_40/while/lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_40/while/lstm_cell_40/dropout_3/MulЕ
*lstm_40/while/lstm_cell_40/dropout_3/ShapeShape-lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_40/while/lstm_cell_40/dropout_3/ShapeЈ
Alstm_40/while/lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_40/while/lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЏБљ2C
Alstm_40/while/lstm_cell_40/dropout_3/random_uniform/RandomUniformЏ
3lstm_40/while/lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_40/while/lstm_cell_40/dropout_3/GreaterEqual/yВ
1lstm_40/while/lstm_cell_40/dropout_3/GreaterEqualGreaterEqualJlstm_40/while/lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:0<lstm_40/while/lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_40/while/lstm_cell_40/dropout_3/GreaterEqualж
)lstm_40/while/lstm_cell_40/dropout_3/CastCast5lstm_40/while/lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_40/while/lstm_cell_40/dropout_3/Castю
*lstm_40/while/lstm_cell_40/dropout_3/Mul_1Mul,lstm_40/while/lstm_cell_40/dropout_3/Mul:z:0-lstm_40/while/lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_40/while/lstm_cell_40/dropout_3/Mul_1
*lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_40/while/lstm_cell_40/split/split_dimо
/lstm_40/while/lstm_cell_40/split/ReadVariableOpReadVariableOp:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_40/while/lstm_cell_40/split/ReadVariableOp
 lstm_40/while/lstm_cell_40/splitSplit3lstm_40/while/lstm_cell_40/split/split_dim:output:07lstm_40/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_40/while/lstm_cell_40/splitч
!lstm_40/while/lstm_cell_40/MatMulMatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_40/while/lstm_cell_40/MatMulы
#lstm_40/while/lstm_cell_40/MatMul_1MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_1ы
#lstm_40/while/lstm_cell_40/MatMul_2MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_2ы
#lstm_40/while/lstm_cell_40/MatMul_3MatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_3
,lstm_40/while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_40/while/lstm_cell_40/split_1/split_dimр
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp
"lstm_40/while/lstm_cell_40/split_1Split5lstm_40/while/lstm_cell_40/split_1/split_dim:output:09lstm_40/while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_40/while/lstm_cell_40/split_1п
"lstm_40/while/lstm_cell_40/BiasAddBiasAdd+lstm_40/while/lstm_cell_40/MatMul:product:0+lstm_40/while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/while/lstm_cell_40/BiasAddх
$lstm_40/while/lstm_cell_40/BiasAdd_1BiasAdd-lstm_40/while/lstm_cell_40/MatMul_1:product:0+lstm_40/while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_1х
$lstm_40/while/lstm_cell_40/BiasAdd_2BiasAdd-lstm_40/while/lstm_cell_40/MatMul_2:product:0+lstm_40/while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_2х
$lstm_40/while/lstm_cell_40/BiasAdd_3BiasAdd-lstm_40/while/lstm_cell_40/MatMul_3:product:0+lstm_40/while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/BiasAdd_3Ф
lstm_40/while/lstm_cell_40/mulMullstm_40_while_placeholder_2,lstm_40/while/lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/while/lstm_cell_40/mulЪ
 lstm_40/while/lstm_cell_40/mul_1Mullstm_40_while_placeholder_2.lstm_40/while/lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_1Ъ
 lstm_40/while/lstm_cell_40/mul_2Mullstm_40_while_placeholder_2.lstm_40/while/lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_2Ъ
 lstm_40/while/lstm_cell_40/mul_3Mullstm_40_while_placeholder_2.lstm_40/while/lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_3Ь
)lstm_40/while/lstm_cell_40/ReadVariableOpReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_40/while/lstm_cell_40/ReadVariableOpБ
.lstm_40/while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_40/while/lstm_cell_40/strided_slice/stackЕ
0lstm_40/while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_40/while/lstm_cell_40/strided_slice/stack_1Е
0lstm_40/while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_40/while/lstm_cell_40/strided_slice/stack_2
(lstm_40/while/lstm_cell_40/strided_sliceStridedSlice1lstm_40/while/lstm_cell_40/ReadVariableOp:value:07lstm_40/while/lstm_cell_40/strided_slice/stack:output:09lstm_40/while/lstm_cell_40/strided_slice/stack_1:output:09lstm_40/while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_40/while/lstm_cell_40/strided_sliceн
#lstm_40/while/lstm_cell_40/MatMul_4MatMul"lstm_40/while/lstm_cell_40/mul:z:01lstm_40/while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_4з
lstm_40/while/lstm_cell_40/addAddV2+lstm_40/while/lstm_cell_40/BiasAdd:output:0-lstm_40/while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/while/lstm_cell_40/addЉ
"lstm_40/while/lstm_cell_40/SigmoidSigmoid"lstm_40/while/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_40/while/lstm_cell_40/Sigmoidа
+lstm_40/while/lstm_cell_40/ReadVariableOp_1ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_1Е
0lstm_40/while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_40/while/lstm_cell_40/strided_slice_1/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_1/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_1StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_1:value:09lstm_40/while/lstm_cell_40/strided_slice_1/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_1/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_1с
#lstm_40/while/lstm_cell_40/MatMul_5MatMul$lstm_40/while/lstm_cell_40/mul_1:z:03lstm_40/while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_5н
 lstm_40/while/lstm_cell_40/add_1AddV2-lstm_40/while/lstm_cell_40/BiasAdd_1:output:0-lstm_40/while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_1Џ
$lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid$lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/Sigmoid_1Ф
 lstm_40/while/lstm_cell_40/mul_4Mul(lstm_40/while/lstm_cell_40/Sigmoid_1:y:0lstm_40_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_4а
+lstm_40/while/lstm_cell_40/ReadVariableOp_2ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_2Е
0lstm_40/while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_40/while/lstm_cell_40/strided_slice_2/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_2/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_2StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_2:value:09lstm_40/while/lstm_cell_40/strided_slice_2/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_2/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_2с
#lstm_40/while/lstm_cell_40/MatMul_6MatMul$lstm_40/while/lstm_cell_40/mul_2:z:03lstm_40/while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_6н
 lstm_40/while/lstm_cell_40/add_2AddV2-lstm_40/while/lstm_cell_40/BiasAdd_2:output:0-lstm_40/while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_2Ђ
lstm_40/while/lstm_cell_40/ReluRelu$lstm_40/while/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_40/while/lstm_cell_40/Reluд
 lstm_40/while/lstm_cell_40/mul_5Mul&lstm_40/while/lstm_cell_40/Sigmoid:y:0-lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_5Ы
 lstm_40/while/lstm_cell_40/add_3AddV2$lstm_40/while/lstm_cell_40/mul_4:z:0$lstm_40/while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_3а
+lstm_40/while/lstm_cell_40/ReadVariableOp_3ReadVariableOp4lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_40/while/lstm_cell_40/ReadVariableOp_3Е
0lstm_40/while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_40/while/lstm_cell_40/strided_slice_3/stackЙ
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_1Й
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_40/while/lstm_cell_40/strided_slice_3/stack_2Њ
*lstm_40/while/lstm_cell_40/strided_slice_3StridedSlice3lstm_40/while/lstm_cell_40/ReadVariableOp_3:value:09lstm_40/while/lstm_cell_40/strided_slice_3/stack:output:0;lstm_40/while/lstm_cell_40/strided_slice_3/stack_1:output:0;lstm_40/while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_40/while/lstm_cell_40/strided_slice_3с
#lstm_40/while/lstm_cell_40/MatMul_7MatMul$lstm_40/while/lstm_cell_40/mul_3:z:03lstm_40/while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_40/while/lstm_cell_40/MatMul_7н
 lstm_40/while/lstm_cell_40/add_4AddV2-lstm_40/while/lstm_cell_40/BiasAdd_3:output:0-lstm_40/while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/add_4Џ
$lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid$lstm_40/while/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_40/while/lstm_cell_40/Sigmoid_2І
!lstm_40/while/lstm_cell_40/Relu_1Relu$lstm_40/while/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_40/while/lstm_cell_40/Relu_1и
 lstm_40/while/lstm_cell_40/mul_6Mul(lstm_40/while/lstm_cell_40/Sigmoid_2:y:0/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_40/while/lstm_cell_40/mul_6
2lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_40_while_placeholder_1lstm_40_while_placeholder$lstm_40/while/lstm_cell_40/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_40/while/TensorArrayV2Write/TensorListSetIteml
lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_40/while/add/y
lstm_40/while/addAddV2lstm_40_while_placeholderlstm_40/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_40/while/addp
lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_40/while/add_1/y
lstm_40/while/add_1AddV2(lstm_40_while_lstm_40_while_loop_counterlstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_40/while/add_1
lstm_40/while/IdentityIdentitylstm_40/while/add_1:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/IdentityІ
lstm_40/while/Identity_1Identity.lstm_40_while_lstm_40_while_maximum_iterations^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_1
lstm_40/while/Identity_2Identitylstm_40/while/add:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_2К
lstm_40/while/Identity_3IdentityBlstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_40/while/NoOp*
T0*
_output_shapes
: 2
lstm_40/while/Identity_3­
lstm_40/while/Identity_4Identity$lstm_40/while/lstm_cell_40/mul_6:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/while/Identity_4­
lstm_40/while/Identity_5Identity$lstm_40/while/lstm_cell_40/add_3:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/while/Identity_5
lstm_40/while/NoOpNoOp*^lstm_40/while/lstm_cell_40/ReadVariableOp,^lstm_40/while/lstm_cell_40/ReadVariableOp_1,^lstm_40/while/lstm_cell_40/ReadVariableOp_2,^lstm_40/while/lstm_cell_40/ReadVariableOp_30^lstm_40/while/lstm_cell_40/split/ReadVariableOp2^lstm_40/while/lstm_cell_40/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_40/while/NoOp"9
lstm_40_while_identitylstm_40/while/Identity:output:0"=
lstm_40_while_identity_1!lstm_40/while/Identity_1:output:0"=
lstm_40_while_identity_2!lstm_40/while/Identity_2:output:0"=
lstm_40_while_identity_3!lstm_40/while/Identity_3:output:0"=
lstm_40_while_identity_4!lstm_40/while/Identity_4:output:0"=
lstm_40_while_identity_5!lstm_40/while/Identity_5:output:0"P
%lstm_40_while_lstm_40_strided_slice_1'lstm_40_while_lstm_40_strided_slice_1_0"j
2lstm_40_while_lstm_cell_40_readvariableop_resource4lstm_40_while_lstm_cell_40_readvariableop_resource_0"z
:lstm_40_while_lstm_cell_40_split_1_readvariableop_resource<lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0"v
8lstm_40_while_lstm_cell_40_split_readvariableop_resource:lstm_40_while_lstm_cell_40_split_readvariableop_resource_0"Ш
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_40/while/lstm_cell_40/ReadVariableOp)lstm_40/while/lstm_cell_40/ReadVariableOp2Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_1+lstm_40/while/lstm_cell_40/ReadVariableOp_12Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_2+lstm_40/while/lstm_cell_40/ReadVariableOp_22Z
+lstm_40/while/lstm_cell_40/ReadVariableOp_3+lstm_40/while/lstm_cell_40/ReadVariableOp_32b
/lstm_40/while/lstm_cell_40/split/ReadVariableOp/lstm_40/while/lstm_cell_40/split/ReadVariableOp2f
1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp1lstm_40/while/lstm_cell_40/split_1/ReadVariableOp: 
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
ј+
Е
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383819
input_17"
lstm_40_1383788:	
lstm_40_1383790:	"
lstm_40_1383792:	 "
dense_48_1383795:  
dense_48_1383797: "
dense_49_1383800: 
dense_49_1383802:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂlstm_40/StatefulPartitionedCallЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЇ
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinput_17lstm_40_1383788lstm_40_1383790lstm_40_1383792*
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13836512!
lstm_40/StatefulPartitionedCallЙ
 dense_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_48_1383795dense_48_1383797*
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
E__inference_dense_48_layer_call_and_return_conditional_losses_13832322"
 dense_48/StatefulPartitionedCallК
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1383800dense_49_1383802*
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
E__inference_dense_49_layer_call_and_return_conditional_losses_13832542"
 dense_49/StatefulPartitionedCall
reshape_24/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_13832732
reshape_24/PartitionedCallЯ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_40_1383788*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЏ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_49_1383802*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mul
IdentityIdentity#reshape_24/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall0^dense_49/bias/Regularizer/Square/ReadVariableOp ^lstm_40/StatefulPartitionedCall>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_17
к
Ш
while_cond_1385211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1385211___redundant_placeholder05
1while_while_cond_1385211___redundant_placeholder15
1while_while_cond_1385211___redundant_placeholder25
1while_while_cond_1385211___redundant_placeholder3
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
while_cond_1383079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1383079___redundant_placeholder05
1while_while_cond_1383079___redundant_placeholder15
1while_while_cond_1383079___redundant_placeholder25
1while_while_cond_1383079___redundant_placeholder3
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
ђ+
Г
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383288

inputs"
lstm_40_1383214:	
lstm_40_1383216:	"
lstm_40_1383218:	 "
dense_48_1383233:  
dense_48_1383235: "
dense_49_1383255: 
dense_49_1383257:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂlstm_40/StatefulPartitionedCallЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЅ
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinputslstm_40_1383214lstm_40_1383216lstm_40_1383218*
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13832132!
lstm_40/StatefulPartitionedCallЙ
 dense_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_48_1383233dense_48_1383235*
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
E__inference_dense_48_layer_call_and_return_conditional_losses_13832322"
 dense_48/StatefulPartitionedCallК
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1383255dense_49_1383257*
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
E__inference_dense_49_layer_call_and_return_conditional_losses_13832542"
 dense_49/StatefulPartitionedCall
reshape_24/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_13832732
reshape_24/PartitionedCallЯ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_40_1383214*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЏ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_49_1383257*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mul
IdentityIdentity#reshape_24/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall0^dense_49/bias/Regularizer/Square/ReadVariableOp ^lstm_40/StatefulPartitionedCall>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Ѕ	
while_body_1383080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Ѕ
while/lstm_cell_40/mulMulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЉ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Љ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Љ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2%while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
ф	
Ј
/__inference_sequential_16_layer_call_fn_1383896

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_13837152
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
к
Ш
while_cond_1382317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1382317___redundant_placeholder05
1while_while_cond_1382317___redundant_placeholder15
1while_while_cond_1382317___redundant_placeholder25
1while_while_cond_1382317___redundant_placeholder3
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385819

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2,
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

c
G__inference_reshape_24_layer_call_and_return_conditional_losses_1385716

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
И
ї
.__inference_lstm_cell_40_layer_call_fn_1385966

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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13825372
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
Й
Љ
(sequential_16_lstm_40_while_body_1382031H
Dsequential_16_lstm_40_while_sequential_16_lstm_40_while_loop_counterN
Jsequential_16_lstm_40_while_sequential_16_lstm_40_while_maximum_iterations+
'sequential_16_lstm_40_while_placeholder-
)sequential_16_lstm_40_while_placeholder_1-
)sequential_16_lstm_40_while_placeholder_2-
)sequential_16_lstm_40_while_placeholder_3G
Csequential_16_lstm_40_while_sequential_16_lstm_40_strided_slice_1_0
sequential_16_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_16_lstm_40_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_16_lstm_40_while_lstm_cell_40_split_readvariableop_resource_0:	Y
Jsequential_16_lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0:	U
Bsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0:	 (
$sequential_16_lstm_40_while_identity*
&sequential_16_lstm_40_while_identity_1*
&sequential_16_lstm_40_while_identity_2*
&sequential_16_lstm_40_while_identity_3*
&sequential_16_lstm_40_while_identity_4*
&sequential_16_lstm_40_while_identity_5E
Asequential_16_lstm_40_while_sequential_16_lstm_40_strided_slice_1
}sequential_16_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_16_lstm_40_tensorarrayunstack_tensorlistfromtensorY
Fsequential_16_lstm_40_while_lstm_cell_40_split_readvariableop_resource:	W
Hsequential_16_lstm_40_while_lstm_cell_40_split_1_readvariableop_resource:	S
@sequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource:	 Ђ7sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOpЂ9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_1Ђ9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_2Ђ9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_3Ђ=sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOpЂ?sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOpя
Msequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2O
Msequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeз
?sequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_16_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_16_lstm_40_tensorarrayunstack_tensorlistfromtensor_0'sequential_16_lstm_40_while_placeholderVsequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02A
?sequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItemЭ
8sequential_16/lstm_40/while/lstm_cell_40/ones_like/ShapeShape)sequential_16_lstm_40_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_16/lstm_40/while/lstm_cell_40/ones_like/ShapeЙ
8sequential_16/lstm_40/while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_16/lstm_40/while/lstm_cell_40/ones_like/ConstЈ
2sequential_16/lstm_40/while/lstm_cell_40/ones_likeFillAsequential_16/lstm_40/while/lstm_cell_40/ones_like/Shape:output:0Asequential_16/lstm_40/while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/ones_likeЖ
8sequential_16/lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_16/lstm_40/while/lstm_cell_40/split/split_dim
=sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOpReadVariableOpHsequential_16_lstm_40_while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02?
=sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOpЫ
.sequential_16/lstm_40/while/lstm_cell_40/splitSplitAsequential_16/lstm_40/while/lstm_cell_40/split/split_dim:output:0Esequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split20
.sequential_16/lstm_40/while/lstm_cell_40/split
/sequential_16/lstm_40/while/lstm_cell_40/MatMulMatMulFsequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_16/lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_16/lstm_40/while/lstm_cell_40/MatMulЃ
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_1MatMulFsequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_16/lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_1Ѓ
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_2MatMulFsequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_16/lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_2Ѓ
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_3MatMulFsequential_16/lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_16/lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_3К
:sequential_16/lstm_40/while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_16/lstm_40/while/lstm_cell_40/split_1/split_dim
?sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOpReadVariableOpJsequential_16_lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOpУ
0sequential_16/lstm_40/while/lstm_cell_40/split_1SplitCsequential_16/lstm_40/while/lstm_cell_40/split_1/split_dim:output:0Gsequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split22
0sequential_16/lstm_40/while/lstm_cell_40/split_1
0sequential_16/lstm_40/while/lstm_cell_40/BiasAddBiasAdd9sequential_16/lstm_40/while/lstm_cell_40/MatMul:product:09sequential_16/lstm_40/while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_16/lstm_40/while/lstm_cell_40/BiasAdd
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_1BiasAdd;sequential_16/lstm_40/while/lstm_cell_40/MatMul_1:product:09sequential_16/lstm_40/while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_1
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_2BiasAdd;sequential_16/lstm_40/while/lstm_cell_40/MatMul_2:product:09sequential_16/lstm_40/while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_2
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_3BiasAdd;sequential_16/lstm_40/while/lstm_cell_40/MatMul_3:product:09sequential_16/lstm_40/while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_3§
,sequential_16/lstm_40/while/lstm_cell_40/mulMul)sequential_16_lstm_40_while_placeholder_2;sequential_16/lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/while/lstm_cell_40/mul
.sequential_16/lstm_40/while/lstm_cell_40/mul_1Mul)sequential_16_lstm_40_while_placeholder_2;sequential_16/lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_1
.sequential_16/lstm_40/while/lstm_cell_40/mul_2Mul)sequential_16_lstm_40_while_placeholder_2;sequential_16/lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_2
.sequential_16/lstm_40/while/lstm_cell_40/mul_3Mul)sequential_16_lstm_40_while_placeholder_2;sequential_16/lstm_40/while/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_3і
7sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOpReadVariableOpBsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype029
7sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOpЭ
<sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stackб
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_1б
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_2ђ
6sequential_16/lstm_40/while/lstm_cell_40/strided_sliceStridedSlice?sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp:value:0Esequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack:output:0Gsequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_1:output:0Gsequential_16/lstm_40/while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_16/lstm_40/while/lstm_cell_40/strided_slice
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_4MatMul0sequential_16/lstm_40/while/lstm_cell_40/mul:z:0?sequential_16/lstm_40/while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_4
,sequential_16/lstm_40/while/lstm_cell_40/addAddV29sequential_16/lstm_40/while/lstm_cell_40/BiasAdd:output:0;sequential_16/lstm_40/while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/while/lstm_cell_40/addг
0sequential_16/lstm_40/while/lstm_cell_40/SigmoidSigmoid0sequential_16/lstm_40/while/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_16/lstm_40/while/lstm_cell_40/Sigmoidњ
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_1ReadVariableOpBsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_1б
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stackе
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_1е
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_2ў
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1StridedSliceAsequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_1:value:0Gsequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_1:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_1
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_5MatMul2sequential_16/lstm_40/while/lstm_cell_40/mul_1:z:0Asequential_16/lstm_40/while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_5
.sequential_16/lstm_40/while/lstm_cell_40/add_1AddV2;sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_1:output:0;sequential_16/lstm_40/while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/add_1й
2sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid2sequential_16/lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_1ќ
.sequential_16/lstm_40/while/lstm_cell_40/mul_4Mul6sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_1:y:0)sequential_16_lstm_40_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_4њ
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_2ReadVariableOpBsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_2б
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stackе
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_1е
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_2ў
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2StridedSliceAsequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_2:value:0Gsequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_1:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_2
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_6MatMul2sequential_16/lstm_40/while/lstm_cell_40/mul_2:z:0Asequential_16/lstm_40/while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_6
.sequential_16/lstm_40/while/lstm_cell_40/add_2AddV2;sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_2:output:0;sequential_16/lstm_40/while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/add_2Ь
-sequential_16/lstm_40/while/lstm_cell_40/ReluRelu2sequential_16/lstm_40/while/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_16/lstm_40/while/lstm_cell_40/Relu
.sequential_16/lstm_40/while/lstm_cell_40/mul_5Mul4sequential_16/lstm_40/while/lstm_cell_40/Sigmoid:y:0;sequential_16/lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_5
.sequential_16/lstm_40/while/lstm_cell_40/add_3AddV22sequential_16/lstm_40/while/lstm_cell_40/mul_4:z:02sequential_16/lstm_40/while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/add_3њ
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_3ReadVariableOpBsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_3б
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2@
>sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stackе
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_1е
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_2ў
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3StridedSliceAsequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_3:value:0Gsequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_1:output:0Isequential_16/lstm_40/while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_16/lstm_40/while/lstm_cell_40/strided_slice_3
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_7MatMul2sequential_16/lstm_40/while/lstm_cell_40/mul_3:z:0Asequential_16/lstm_40/while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_16/lstm_40/while/lstm_cell_40/MatMul_7
.sequential_16/lstm_40/while/lstm_cell_40/add_4AddV2;sequential_16/lstm_40/while/lstm_cell_40/BiasAdd_3:output:0;sequential_16/lstm_40/while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/add_4й
2sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid2sequential_16/lstm_40/while/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_2а
/sequential_16/lstm_40/while/lstm_cell_40/Relu_1Relu2sequential_16/lstm_40/while/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_16/lstm_40/while/lstm_cell_40/Relu_1
.sequential_16/lstm_40/while/lstm_cell_40/mul_6Mul6sequential_16/lstm_40/while/lstm_cell_40/Sigmoid_2:y:0=sequential_16/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_16/lstm_40/while/lstm_cell_40/mul_6Ю
@sequential_16/lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_16_lstm_40_while_placeholder_1'sequential_16_lstm_40_while_placeholder2sequential_16/lstm_40/while/lstm_cell_40/mul_6:z:0*
_output_shapes
: *
element_dtype02B
@sequential_16/lstm_40/while/TensorArrayV2Write/TensorListSetItem
!sequential_16/lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_16/lstm_40/while/add/yС
sequential_16/lstm_40/while/addAddV2'sequential_16_lstm_40_while_placeholder*sequential_16/lstm_40/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_16/lstm_40/while/add
#sequential_16/lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_16/lstm_40/while/add_1/yф
!sequential_16/lstm_40/while/add_1AddV2Dsequential_16_lstm_40_while_sequential_16_lstm_40_while_loop_counter,sequential_16/lstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_16/lstm_40/while/add_1У
$sequential_16/lstm_40/while/IdentityIdentity%sequential_16/lstm_40/while/add_1:z:0!^sequential_16/lstm_40/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_16/lstm_40/while/Identityь
&sequential_16/lstm_40/while/Identity_1IdentityJsequential_16_lstm_40_while_sequential_16_lstm_40_while_maximum_iterations!^sequential_16/lstm_40/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_16/lstm_40/while/Identity_1Х
&sequential_16/lstm_40/while/Identity_2Identity#sequential_16/lstm_40/while/add:z:0!^sequential_16/lstm_40/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_16/lstm_40/while/Identity_2ђ
&sequential_16/lstm_40/while/Identity_3IdentityPsequential_16/lstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_16/lstm_40/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_16/lstm_40/while/Identity_3х
&sequential_16/lstm_40/while/Identity_4Identity2sequential_16/lstm_40/while/lstm_cell_40/mul_6:z:0!^sequential_16/lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_16/lstm_40/while/Identity_4х
&sequential_16/lstm_40/while/Identity_5Identity2sequential_16/lstm_40/while/lstm_cell_40/add_3:z:0!^sequential_16/lstm_40/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_16/lstm_40/while/Identity_5і
 sequential_16/lstm_40/while/NoOpNoOp8^sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp:^sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_1:^sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_2:^sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_3>^sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOp@^sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_16/lstm_40/while/NoOp"U
$sequential_16_lstm_40_while_identity-sequential_16/lstm_40/while/Identity:output:0"Y
&sequential_16_lstm_40_while_identity_1/sequential_16/lstm_40/while/Identity_1:output:0"Y
&sequential_16_lstm_40_while_identity_2/sequential_16/lstm_40/while/Identity_2:output:0"Y
&sequential_16_lstm_40_while_identity_3/sequential_16/lstm_40/while/Identity_3:output:0"Y
&sequential_16_lstm_40_while_identity_4/sequential_16/lstm_40/while/Identity_4:output:0"Y
&sequential_16_lstm_40_while_identity_5/sequential_16/lstm_40/while/Identity_5:output:0"
@sequential_16_lstm_40_while_lstm_cell_40_readvariableop_resourceBsequential_16_lstm_40_while_lstm_cell_40_readvariableop_resource_0"
Hsequential_16_lstm_40_while_lstm_cell_40_split_1_readvariableop_resourceJsequential_16_lstm_40_while_lstm_cell_40_split_1_readvariableop_resource_0"
Fsequential_16_lstm_40_while_lstm_cell_40_split_readvariableop_resourceHsequential_16_lstm_40_while_lstm_cell_40_split_readvariableop_resource_0"
Asequential_16_lstm_40_while_sequential_16_lstm_40_strided_slice_1Csequential_16_lstm_40_while_sequential_16_lstm_40_strided_slice_1_0"
}sequential_16_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_16_lstm_40_tensorarrayunstack_tensorlistfromtensorsequential_16_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_16_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2r
7sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp7sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp2v
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_19sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_12v
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_29sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_22v
9sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_39sequential_16/lstm_40/while/lstm_cell_40/ReadVariableOp_32~
=sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOp=sequential_16/lstm_40/while/lstm_cell_40/split/ReadVariableOp2
?sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOp?sequential_16/lstm_40/while/lstm_cell_40/split_1/ReadVariableOp: 
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
кЯ
Ј
D__inference_lstm_40_layer_call_and_return_conditional_losses_1383651

inputs=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileD
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like}
lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout/ConstГ
lstm_cell_40/dropout/MulMullstm_cell_40/ones_like:output:0#lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul
lstm_cell_40/dropout/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout/Shapeј
1lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Зы23
1lstm_cell_40/dropout/random_uniform/RandomUniform
#lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_40/dropout/GreaterEqual/yђ
!lstm_cell_40/dropout/GreaterEqualGreaterEqual:lstm_cell_40/dropout/random_uniform/RandomUniform:output:0,lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_40/dropout/GreaterEqualІ
lstm_cell_40/dropout/CastCast%lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/CastЎ
lstm_cell_40/dropout/Mul_1Mullstm_cell_40/dropout/Mul:z:0lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul_1
lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_1/ConstЙ
lstm_cell_40/dropout_1/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul
lstm_cell_40/dropout_1/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_1/Shapeў
3lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЕШ25
3lstm_cell_40/dropout_1/random_uniform/RandomUniform
%lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_1/GreaterEqual/yњ
#lstm_cell_40/dropout_1/GreaterEqualGreaterEqual<lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_1/GreaterEqualЌ
lstm_cell_40/dropout_1/CastCast'lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/CastЖ
lstm_cell_40/dropout_1/Mul_1Mullstm_cell_40/dropout_1/Mul:z:0lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul_1
lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_2/ConstЙ
lstm_cell_40/dropout_2/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul
lstm_cell_40/dropout_2/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_2/Shape§
3lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Яўi25
3lstm_cell_40/dropout_2/random_uniform/RandomUniform
%lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_2/GreaterEqual/yњ
#lstm_cell_40/dropout_2/GreaterEqualGreaterEqual<lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_2/GreaterEqualЌ
lstm_cell_40/dropout_2/CastCast'lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/CastЖ
lstm_cell_40/dropout_2/Mul_1Mullstm_cell_40/dropout_2/Mul:z:0lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul_1
lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_3/ConstЙ
lstm_cell_40/dropout_3/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul
lstm_cell_40/dropout_3/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_3/Shape§
3lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЏЭ25
3lstm_cell_40/dropout_3/random_uniform/RandomUniform
%lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_3/GreaterEqual/yњ
#lstm_cell_40/dropout_3/GreaterEqualGreaterEqual<lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_3/GreaterEqualЌ
lstm_cell_40/dropout_3/CastCast'lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/CastЖ
lstm_cell_40/dropout_3/Mul_1Mullstm_cell_40/dropout_3/Mul:z:0lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul_1~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0 lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0 lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0 lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1383486*
condR
while_cond_1383485*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_1385486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1385486___redundant_placeholder05
1while_while_cond_1385486___redundant_placeholder15
1while_while_cond_1385486___redundant_placeholder25
1while_while_cond_1385486___redundant_placeholder3
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
Ъ
H
,__inference_reshape_24_layer_call_fn_1385721

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
G__inference_reshape_24_layer_call_and_return_conditional_losses_13832732
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
э
Ј
E__inference_dense_49_layer_call_and_return_conditional_losses_1383254

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_49/bias/Regularizer/Square/ReadVariableOp
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
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_49/bias/Regularizer/Square/ReadVariableOp*"
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
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ъ	
Њ
/__inference_sequential_16_layer_call_fn_1383751
input_17
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_13837152
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
input_17
Ђ
Њ
D__inference_lstm_40_layer_call_and_return_conditional_losses_1384795
inputs_0=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileF
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1384662*
condR
while_cond_1384661*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
н
Ы
__inference_loss_fn_1_1385977Y
Flstm_40_lstm_cell_40_kernel_regularizer_square_readvariableop_resource:	
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_40_lstm_cell_40_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muly
IdentityIdentity/lstm_40/lstm_cell_40/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp
к
Ш
while_cond_1384661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1384661___redundant_placeholder05
1while_while_cond_1384661___redundant_placeholder15
1while_while_cond_1384661___redundant_placeholder25
1while_while_cond_1384661___redundant_placeholder3
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
ѓ

*__inference_dense_49_layer_call_fn_1385703

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
E__inference_dense_49_layer_call_and_return_conditional_losses_13832542
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
сЁ
Ј
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385345

inputs=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileD
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1385212*
condR
while_cond_1385211*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓA
у
 __inference__traced_save_1386084
file_prefix.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_40_lstm_cell_40_kernel_read_readvariableopD
@savev2_lstm_40_lstm_cell_40_recurrent_kernel_read_readvariableop8
4savev2_lstm_40_lstm_cell_40_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_48_kernel_m_read_readvariableop3
/savev2_adam_dense_48_bias_m_read_readvariableop5
1savev2_adam_dense_49_kernel_m_read_readvariableop3
/savev2_adam_dense_49_bias_m_read_readvariableopA
=savev2_adam_lstm_40_lstm_cell_40_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_40_lstm_cell_40_bias_m_read_readvariableop5
1savev2_adam_dense_48_kernel_v_read_readvariableop3
/savev2_adam_dense_48_bias_v_read_readvariableop5
1savev2_adam_dense_49_kernel_v_read_readvariableop3
/savev2_adam_dense_49_bias_v_read_readvariableopA
=savev2_adam_lstm_40_lstm_cell_40_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_40_lstm_cell_40_bias_v_read_readvariableop
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
ShardedFilenameі
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueўBћB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesм
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_40_lstm_cell_40_kernel_read_readvariableop@savev2_lstm_40_lstm_cell_40_recurrent_kernel_read_readvariableop4savev2_lstm_40_lstm_cell_40_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_48_kernel_m_read_readvariableop/savev2_adam_dense_48_bias_m_read_readvariableop1savev2_adam_dense_49_kernel_m_read_readvariableop/savev2_adam_dense_49_bias_m_read_readvariableop=savev2_adam_lstm_40_lstm_cell_40_kernel_m_read_readvariableopGsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_40_lstm_cell_40_bias_m_read_readvariableop1savev2_adam_dense_48_kernel_v_read_readvariableop/savev2_adam_dense_48_bias_v_read_readvariableop1savev2_adam_dense_49_kernel_v_read_readvariableop/savev2_adam_dense_49_bias_v_read_readvariableop=savev2_adam_lstm_40_lstm_cell_40_kernel_v_read_readvariableopGsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_40_lstm_cell_40_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Ы

ш
lstm_40_while_cond_1384005,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3.
*lstm_40_while_less_lstm_40_strided_slice_1E
Alstm_40_while_lstm_40_while_cond_1384005___redundant_placeholder0E
Alstm_40_while_lstm_40_while_cond_1384005___redundant_placeholder1E
Alstm_40_while_lstm_40_while_cond_1384005___redundant_placeholder2E
Alstm_40_while_lstm_40_while_cond_1384005___redundant_placeholder3
lstm_40_while_identity

lstm_40/while/LessLesslstm_40_while_placeholder*lstm_40_while_less_lstm_40_strided_slice_1*
T0*
_output_shapes
: 2
lstm_40/while/Lessu
lstm_40/while/IdentityIdentitylstm_40/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_40/while/Identity"9
lstm_40_while_identitylstm_40/while/Identity:output:0*(
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
э
Ј
E__inference_dense_49_layer_call_and_return_conditional_losses_1385694

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_49/bias/Regularizer/Square/ReadVariableOp
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
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_49/bias/Regularizer/Square/ReadVariableOp*"
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
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
а
Њ
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385102
inputs_0=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileF
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like}
lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout/ConstГ
lstm_cell_40/dropout/MulMullstm_cell_40/ones_like:output:0#lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul
lstm_cell_40/dropout/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout/Shapeї
1lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2асH23
1lstm_cell_40/dropout/random_uniform/RandomUniform
#lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_40/dropout/GreaterEqual/yђ
!lstm_cell_40/dropout/GreaterEqualGreaterEqual:lstm_cell_40/dropout/random_uniform/RandomUniform:output:0,lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_40/dropout/GreaterEqualІ
lstm_cell_40/dropout/CastCast%lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/CastЎ
lstm_cell_40/dropout/Mul_1Mullstm_cell_40/dropout/Mul:z:0lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul_1
lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_1/ConstЙ
lstm_cell_40/dropout_1/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul
lstm_cell_40/dropout_1/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_1/Shape§
3lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2жќU25
3lstm_cell_40/dropout_1/random_uniform/RandomUniform
%lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_1/GreaterEqual/yњ
#lstm_cell_40/dropout_1/GreaterEqualGreaterEqual<lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_1/GreaterEqualЌ
lstm_cell_40/dropout_1/CastCast'lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/CastЖ
lstm_cell_40/dropout_1/Mul_1Mullstm_cell_40/dropout_1/Mul:z:0lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul_1
lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_2/ConstЙ
lstm_cell_40/dropout_2/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul
lstm_cell_40/dropout_2/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_2/Shapeў
3lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Тт25
3lstm_cell_40/dropout_2/random_uniform/RandomUniform
%lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_2/GreaterEqual/yњ
#lstm_cell_40/dropout_2/GreaterEqualGreaterEqual<lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_2/GreaterEqualЌ
lstm_cell_40/dropout_2/CastCast'lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/CastЖ
lstm_cell_40/dropout_2/Mul_1Mullstm_cell_40/dropout_2/Mul:z:0lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul_1
lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_3/ConstЙ
lstm_cell_40/dropout_3/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul
lstm_cell_40/dropout_3/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_3/Shape§
3lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ёы!25
3lstm_cell_40/dropout_3/random_uniform/RandomUniform
%lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_3/GreaterEqual/yњ
#lstm_cell_40/dropout_3/GreaterEqualGreaterEqual<lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_3/GreaterEqualЌ
lstm_cell_40/dropout_3/CastCast'lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/CastЖ
lstm_cell_40/dropout_3/Mul_1Mullstm_cell_40/dropout_3/Mul:z:0lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul_1~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0 lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0 lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0 lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1384937*
condR
while_cond_1384936*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ћВ
Ѕ	
while_body_1383486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_40_split_readvariableop_resource_0:	C
4while_lstm_cell_40_split_1_readvariableop_resource_0:	?
,while_lstm_cell_40_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_40_split_readvariableop_resource:	A
2while_lstm_cell_40_split_1_readvariableop_resource:	=
*while_lstm_cell_40_readvariableop_resource:	 Ђ!while/lstm_cell_40/ReadVariableOpЂ#while/lstm_cell_40/ReadVariableOp_1Ђ#while/lstm_cell_40/ReadVariableOp_2Ђ#while/lstm_cell_40/ReadVariableOp_3Ђ'while/lstm_cell_40/split/ReadVariableOpЂ)while/lstm_cell_40/split_1/ReadVariableOpУ
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
"while/lstm_cell_40/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_40/ones_like/Shape
"while/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_40/ones_like/Constа
while/lstm_cell_40/ones_likeFill+while/lstm_cell_40/ones_like/Shape:output:0+while/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ones_like
 while/lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_40/dropout/ConstЫ
while/lstm_cell_40/dropout/MulMul%while/lstm_cell_40/ones_like:output:0)while/lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_40/dropout/Mul
 while/lstm_cell_40/dropout/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_40/dropout/Shape
7while/lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЧИu29
7while/lstm_cell_40/dropout/random_uniform/RandomUniform
)while/lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_40/dropout/GreaterEqual/y
'while/lstm_cell_40/dropout/GreaterEqualGreaterEqual@while/lstm_cell_40/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_40/dropout/GreaterEqualИ
while/lstm_cell_40/dropout/CastCast+while/lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_40/dropout/CastЦ
 while/lstm_cell_40/dropout/Mul_1Mul"while/lstm_cell_40/dropout/Mul:z:0#while/lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout/Mul_1
"while/lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_1/Constб
 while/lstm_cell_40/dropout_1/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_1/Mul
"while/lstm_cell_40/dropout_1/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_1/Shape
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2цЩЃ2;
9while/lstm_cell_40/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_1/GreaterEqual/y
)while/lstm_cell_40/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_1/GreaterEqualО
!while/lstm_cell_40/dropout_1/CastCast-while/lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_1/CastЮ
"while/lstm_cell_40/dropout_1/Mul_1Mul$while/lstm_cell_40/dropout_1/Mul:z:0%while/lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_1/Mul_1
"while/lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_2/Constб
 while/lstm_cell_40/dropout_2/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_2/Mul
"while/lstm_cell_40/dropout_2/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_2/Shape
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЋЁб2;
9while/lstm_cell_40/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_2/GreaterEqual/y
)while/lstm_cell_40/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_2/GreaterEqualО
!while/lstm_cell_40/dropout_2/CastCast-while/lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_2/CastЮ
"while/lstm_cell_40/dropout_2/Mul_1Mul$while/lstm_cell_40/dropout_2/Mul:z:0%while/lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_2/Mul_1
"while/lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_40/dropout_3/Constб
 while/lstm_cell_40/dropout_3/MulMul%while/lstm_cell_40/ones_like:output:0+while/lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_40/dropout_3/Mul
"while/lstm_cell_40/dropout_3/ShapeShape%while/lstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_40/dropout_3/Shape
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЅЧЧ2;
9while/lstm_cell_40/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_40/dropout_3/GreaterEqual/y
)while/lstm_cell_40/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_40/dropout_3/GreaterEqualО
!while/lstm_cell_40/dropout_3/CastCast-while/lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_40/dropout_3/CastЮ
"while/lstm_cell_40/dropout_3/Mul_1Mul$while/lstm_cell_40/dropout_3/Mul:z:0%while/lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_40/dropout_3/Mul_1
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_40/split/split_dimЦ
'while/lstm_cell_40/split/ReadVariableOpReadVariableOp2while_lstm_cell_40_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_40/split/ReadVariableOpѓ
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0/while/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_40/splitЧ
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMulЫ
while/lstm_cell_40/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_1Ы
while/lstm_cell_40/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_2Ы
while/lstm_cell_40/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_3
$while/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_40/split_1/split_dimШ
)while/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_40_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_40/split_1/ReadVariableOpы
while/lstm_cell_40/split_1Split-while/lstm_cell_40/split_1/split_dim:output:01while/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_40/split_1П
while/lstm_cell_40/BiasAddBiasAdd#while/lstm_cell_40/MatMul:product:0#while/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAddХ
while/lstm_cell_40/BiasAdd_1BiasAdd%while/lstm_cell_40/MatMul_1:product:0#while/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_1Х
while/lstm_cell_40/BiasAdd_2BiasAdd%while/lstm_cell_40/MatMul_2:product:0#while/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_2Х
while/lstm_cell_40/BiasAdd_3BiasAdd%while/lstm_cell_40/MatMul_3:product:0#while/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/BiasAdd_3Є
while/lstm_cell_40/mulMulwhile_placeholder_2$while/lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mulЊ
while/lstm_cell_40/mul_1Mulwhile_placeholder_2&while/lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_1Њ
while/lstm_cell_40/mul_2Mulwhile_placeholder_2&while/lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_2Њ
while/lstm_cell_40/mul_3Mulwhile_placeholder_2&while/lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_3Д
!while/lstm_cell_40/ReadVariableOpReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_40/ReadVariableOpЁ
&while/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_40/strided_slice/stackЅ
(while/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice/stack_1Ѕ
(while/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_40/strided_slice/stack_2ю
 while/lstm_cell_40/strided_sliceStridedSlice)while/lstm_cell_40/ReadVariableOp:value:0/while/lstm_cell_40/strided_slice/stack:output:01while/lstm_cell_40/strided_slice/stack_1:output:01while/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_40/strided_sliceН
while/lstm_cell_40/MatMul_4MatMulwhile/lstm_cell_40/mul:z:0)while/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_4З
while/lstm_cell_40/addAddV2#while/lstm_cell_40/BiasAdd:output:0%while/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add
while/lstm_cell_40/SigmoidSigmoidwhile/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/SigmoidИ
#while/lstm_cell_40/ReadVariableOp_1ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_1Ѕ
(while/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_40/strided_slice_1/stackЉ
*while/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_40/strided_slice_1/stack_1Љ
*while/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_1/stack_2њ
"while/lstm_cell_40/strided_slice_1StridedSlice+while/lstm_cell_40/ReadVariableOp_1:value:01while/lstm_cell_40/strided_slice_1/stack:output:03while/lstm_cell_40/strided_slice_1/stack_1:output:03while/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_1С
while/lstm_cell_40/MatMul_5MatMulwhile/lstm_cell_40/mul_1:z:0+while/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_5Н
while/lstm_cell_40/add_1AddV2%while/lstm_cell_40/BiasAdd_1:output:0%while/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_1
while/lstm_cell_40/Sigmoid_1Sigmoidwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_1Є
while/lstm_cell_40/mul_4Mul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_4И
#while/lstm_cell_40/ReadVariableOp_2ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_2Ѕ
(while/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_40/strided_slice_2/stackЉ
*while/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_40/strided_slice_2/stack_1Љ
*while/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_2/stack_2њ
"while/lstm_cell_40/strided_slice_2StridedSlice+while/lstm_cell_40/ReadVariableOp_2:value:01while/lstm_cell_40/strided_slice_2/stack:output:03while/lstm_cell_40/strided_slice_2/stack_1:output:03while/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_2С
while/lstm_cell_40/MatMul_6MatMulwhile/lstm_cell_40/mul_2:z:0+while/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_6Н
while/lstm_cell_40/add_2AddV2%while/lstm_cell_40/BiasAdd_2:output:0%while/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_2
while/lstm_cell_40/ReluReluwhile/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/ReluД
while/lstm_cell_40/mul_5Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_5Ћ
while/lstm_cell_40/add_3AddV2while/lstm_cell_40/mul_4:z:0while/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_3И
#while/lstm_cell_40/ReadVariableOp_3ReadVariableOp,while_lstm_cell_40_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_40/ReadVariableOp_3Ѕ
(while/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_40/strided_slice_3/stackЉ
*while/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_40/strided_slice_3/stack_1Љ
*while/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_40/strided_slice_3/stack_2њ
"while/lstm_cell_40/strided_slice_3StridedSlice+while/lstm_cell_40/ReadVariableOp_3:value:01while/lstm_cell_40/strided_slice_3/stack:output:03while/lstm_cell_40/strided_slice_3/stack_1:output:03while/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_40/strided_slice_3С
while/lstm_cell_40/MatMul_7MatMulwhile/lstm_cell_40/mul_3:z:0+while/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/MatMul_7Н
while/lstm_cell_40/add_4AddV2%while/lstm_cell_40/BiasAdd_3:output:0%while/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/add_4
while/lstm_cell_40/Sigmoid_2Sigmoidwhile/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Sigmoid_2
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/Relu_1И
while/lstm_cell_40/mul_6Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_40/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_40/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_40/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_40/ReadVariableOp$^while/lstm_cell_40/ReadVariableOp_1$^while/lstm_cell_40/ReadVariableOp_2$^while/lstm_cell_40/ReadVariableOp_3(^while/lstm_cell_40/split/ReadVariableOp*^while/lstm_cell_40/split_1/ReadVariableOp*"
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
*while_lstm_cell_40_readvariableop_resource,while_lstm_cell_40_readvariableop_resource_0"j
2while_lstm_cell_40_split_1_readvariableop_resource4while_lstm_cell_40_split_1_readvariableop_resource_0"f
0while_lstm_cell_40_split_readvariableop_resource2while_lstm_cell_40_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_40/ReadVariableOp!while/lstm_cell_40/ReadVariableOp2J
#while/lstm_cell_40/ReadVariableOp_1#while/lstm_cell_40/ReadVariableOp_12J
#while/lstm_cell_40/ReadVariableOp_2#while/lstm_cell_40/ReadVariableOp_22J
#while/lstm_cell_40/ReadVariableOp_3#while/lstm_cell_40/ReadVariableOp_32R
'while/lstm_cell_40/split/ReadVariableOp'while/lstm_cell_40/split/ReadVariableOp2V
)while/lstm_cell_40/split_1/ReadVariableOp)while/lstm_cell_40/split_1/ReadVariableOp: 
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
ф	
Ј
/__inference_sequential_16_layer_call_fn_1383877

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_13832882
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
И
ї
.__inference_lstm_cell_40_layer_call_fn_1385949

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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13823042
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
__inference_loss_fn_0_1385732F
8dense_49_bias_regularizer_square_readvariableop_resource:
identityЂ/dense_49/bias/Regularizer/Square/ReadVariableOpз
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_49_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mulk
IdentityIdentity!dense_49/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense_49/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp
с

J__inference_sequential_16_layer_call_and_return_conditional_losses_1384167

inputsE
2lstm_40_lstm_cell_40_split_readvariableop_resource:	C
4lstm_40_lstm_cell_40_split_1_readvariableop_resource:	?
,lstm_40_lstm_cell_40_readvariableop_resource:	 9
'dense_48_matmul_readvariableop_resource:  6
(dense_48_biasadd_readvariableop_resource: 9
'dense_49_matmul_readvariableop_resource: 6
(dense_49_biasadd_readvariableop_resource:
identityЂdense_48/BiasAdd/ReadVariableOpЂdense_48/MatMul/ReadVariableOpЂdense_49/BiasAdd/ReadVariableOpЂdense_49/MatMul/ReadVariableOpЂ/dense_49/bias/Regularizer/Square/ReadVariableOpЂ#lstm_40/lstm_cell_40/ReadVariableOpЂ%lstm_40/lstm_cell_40/ReadVariableOp_1Ђ%lstm_40/lstm_cell_40/ReadVariableOp_2Ђ%lstm_40/lstm_cell_40/ReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_40/lstm_cell_40/split/ReadVariableOpЂ+lstm_40/lstm_cell_40/split_1/ReadVariableOpЂlstm_40/whileT
lstm_40/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_40/Shape
lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice/stack
lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_40/strided_slice/stack_1
lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_40/strided_slice/stack_2
lstm_40/strided_sliceStridedSlicelstm_40/Shape:output:0$lstm_40/strided_slice/stack:output:0&lstm_40/strided_slice/stack_1:output:0&lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_40/strided_slicel
lstm_40/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros/mul/y
lstm_40/zeros/mulMullstm_40/strided_slice:output:0lstm_40/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros/mulo
lstm_40/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_40/zeros/Less/y
lstm_40/zeros/LessLesslstm_40/zeros/mul:z:0lstm_40/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros/Lessr
lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros/packed/1Ѓ
lstm_40/zeros/packedPacklstm_40/strided_slice:output:0lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_40/zeros/packedo
lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/zeros/Const
lstm_40/zerosFilllstm_40/zeros/packed:output:0lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/zerosp
lstm_40/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros_1/mul/y
lstm_40/zeros_1/mulMullstm_40/strided_slice:output:0lstm_40/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros_1/muls
lstm_40/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_40/zeros_1/Less/y
lstm_40/zeros_1/LessLesslstm_40/zeros_1/mul:z:0lstm_40/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_40/zeros_1/Lessv
lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/zeros_1/packed/1Љ
lstm_40/zeros_1/packedPacklstm_40/strided_slice:output:0!lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_40/zeros_1/packeds
lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/zeros_1/Const
lstm_40/zeros_1Filllstm_40/zeros_1/packed:output:0lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/zeros_1
lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_40/transpose/perm
lstm_40/transpose	Transposeinputslstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_40/transposeg
lstm_40/Shape_1Shapelstm_40/transpose:y:0*
T0*
_output_shapes
:2
lstm_40/Shape_1
lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice_1/stack
lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_1/stack_1
lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_1/stack_2
lstm_40/strided_slice_1StridedSlicelstm_40/Shape_1:output:0&lstm_40/strided_slice_1/stack:output:0(lstm_40/strided_slice_1/stack_1:output:0(lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_40/strided_slice_1
#lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_40/TensorArrayV2/element_shapeв
lstm_40/TensorArrayV2TensorListReserve,lstm_40/TensorArrayV2/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_40/TensorArrayV2Я
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_40/transpose:y:0Flstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_40/TensorArrayUnstack/TensorListFromTensor
lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_40/strided_slice_2/stack
lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_2/stack_1
lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_2/stack_2Ќ
lstm_40/strided_slice_2StridedSlicelstm_40/transpose:y:0&lstm_40/strided_slice_2/stack:output:0(lstm_40/strided_slice_2/stack_1:output:0(lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_40/strided_slice_2
$lstm_40/lstm_cell_40/ones_like/ShapeShapelstm_40/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_40/lstm_cell_40/ones_like/Shape
$lstm_40/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_40/lstm_cell_40/ones_like/Constи
lstm_40/lstm_cell_40/ones_likeFill-lstm_40/lstm_cell_40/ones_like/Shape:output:0-lstm_40/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/ones_like
$lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_40/lstm_cell_40/split/split_dimЪ
)lstm_40/lstm_cell_40/split/ReadVariableOpReadVariableOp2lstm_40_lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_40/lstm_cell_40/split/ReadVariableOpћ
lstm_40/lstm_cell_40/splitSplit-lstm_40/lstm_cell_40/split/split_dim:output:01lstm_40/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_40/lstm_cell_40/splitН
lstm_40/lstm_cell_40/MatMulMatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMulС
lstm_40/lstm_cell_40/MatMul_1MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_1С
lstm_40/lstm_cell_40/MatMul_2MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_2С
lstm_40/lstm_cell_40/MatMul_3MatMul lstm_40/strided_slice_2:output:0#lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_3
&lstm_40/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_40/lstm_cell_40/split_1/split_dimЬ
+lstm_40/lstm_cell_40/split_1/ReadVariableOpReadVariableOp4lstm_40_lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_40/lstm_cell_40/split_1/ReadVariableOpѓ
lstm_40/lstm_cell_40/split_1Split/lstm_40/lstm_cell_40/split_1/split_dim:output:03lstm_40/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_40/lstm_cell_40/split_1Ч
lstm_40/lstm_cell_40/BiasAddBiasAdd%lstm_40/lstm_cell_40/MatMul:product:0%lstm_40/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/BiasAddЭ
lstm_40/lstm_cell_40/BiasAdd_1BiasAdd'lstm_40/lstm_cell_40/MatMul_1:product:0%lstm_40/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_1Э
lstm_40/lstm_cell_40/BiasAdd_2BiasAdd'lstm_40/lstm_cell_40/MatMul_2:product:0%lstm_40/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_2Э
lstm_40/lstm_cell_40/BiasAdd_3BiasAdd'lstm_40/lstm_cell_40/MatMul_3:product:0%lstm_40/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/BiasAdd_3Ў
lstm_40/lstm_cell_40/mulMullstm_40/zeros:output:0'lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mulВ
lstm_40/lstm_cell_40/mul_1Mullstm_40/zeros:output:0'lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_1В
lstm_40/lstm_cell_40/mul_2Mullstm_40/zeros:output:0'lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_2В
lstm_40/lstm_cell_40/mul_3Mullstm_40/zeros:output:0'lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_3И
#lstm_40/lstm_cell_40/ReadVariableOpReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_40/lstm_cell_40/ReadVariableOpЅ
(lstm_40/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_40/lstm_cell_40/strided_slice/stackЉ
*lstm_40/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_40/lstm_cell_40/strided_slice/stack_1Љ
*lstm_40/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_40/lstm_cell_40/strided_slice/stack_2њ
"lstm_40/lstm_cell_40/strided_sliceStridedSlice+lstm_40/lstm_cell_40/ReadVariableOp:value:01lstm_40/lstm_cell_40/strided_slice/stack:output:03lstm_40/lstm_cell_40/strided_slice/stack_1:output:03lstm_40/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_40/lstm_cell_40/strided_sliceХ
lstm_40/lstm_cell_40/MatMul_4MatMullstm_40/lstm_cell_40/mul:z:0+lstm_40/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_4П
lstm_40/lstm_cell_40/addAddV2%lstm_40/lstm_cell_40/BiasAdd:output:0'lstm_40/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add
lstm_40/lstm_cell_40/SigmoidSigmoidlstm_40/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/SigmoidМ
%lstm_40/lstm_cell_40/ReadVariableOp_1ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_1Љ
*lstm_40/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_40/lstm_cell_40/strided_slice_1/stack­
,lstm_40/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_40/lstm_cell_40/strided_slice_1/stack_1­
,lstm_40/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_1/stack_2
$lstm_40/lstm_cell_40/strided_slice_1StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_1:value:03lstm_40/lstm_cell_40/strided_slice_1/stack:output:05lstm_40/lstm_cell_40/strided_slice_1/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_1Щ
lstm_40/lstm_cell_40/MatMul_5MatMullstm_40/lstm_cell_40/mul_1:z:0-lstm_40/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_5Х
lstm_40/lstm_cell_40/add_1AddV2'lstm_40/lstm_cell_40/BiasAdd_1:output:0'lstm_40/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_1
lstm_40/lstm_cell_40/Sigmoid_1Sigmoidlstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/Sigmoid_1Џ
lstm_40/lstm_cell_40/mul_4Mul"lstm_40/lstm_cell_40/Sigmoid_1:y:0lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_4М
%lstm_40/lstm_cell_40/ReadVariableOp_2ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_2Љ
*lstm_40/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_40/lstm_cell_40/strided_slice_2/stack­
,lstm_40/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_40/lstm_cell_40/strided_slice_2/stack_1­
,lstm_40/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_2/stack_2
$lstm_40/lstm_cell_40/strided_slice_2StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_2:value:03lstm_40/lstm_cell_40/strided_slice_2/stack:output:05lstm_40/lstm_cell_40/strided_slice_2/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_2Щ
lstm_40/lstm_cell_40/MatMul_6MatMullstm_40/lstm_cell_40/mul_2:z:0-lstm_40/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_6Х
lstm_40/lstm_cell_40/add_2AddV2'lstm_40/lstm_cell_40/BiasAdd_2:output:0'lstm_40/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_2
lstm_40/lstm_cell_40/ReluRelulstm_40/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/ReluМ
lstm_40/lstm_cell_40/mul_5Mul lstm_40/lstm_cell_40/Sigmoid:y:0'lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_5Г
lstm_40/lstm_cell_40/add_3AddV2lstm_40/lstm_cell_40/mul_4:z:0lstm_40/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_3М
%lstm_40/lstm_cell_40/ReadVariableOp_3ReadVariableOp,lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_40/lstm_cell_40/ReadVariableOp_3Љ
*lstm_40/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_40/lstm_cell_40/strided_slice_3/stack­
,lstm_40/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_40/lstm_cell_40/strided_slice_3/stack_1­
,lstm_40/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_40/lstm_cell_40/strided_slice_3/stack_2
$lstm_40/lstm_cell_40/strided_slice_3StridedSlice-lstm_40/lstm_cell_40/ReadVariableOp_3:value:03lstm_40/lstm_cell_40/strided_slice_3/stack:output:05lstm_40/lstm_cell_40/strided_slice_3/stack_1:output:05lstm_40/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_40/lstm_cell_40/strided_slice_3Щ
lstm_40/lstm_cell_40/MatMul_7MatMullstm_40/lstm_cell_40/mul_3:z:0-lstm_40/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/MatMul_7Х
lstm_40/lstm_cell_40/add_4AddV2'lstm_40/lstm_cell_40/BiasAdd_3:output:0'lstm_40/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/add_4
lstm_40/lstm_cell_40/Sigmoid_2Sigmoidlstm_40/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_40/lstm_cell_40/Sigmoid_2
lstm_40/lstm_cell_40/Relu_1Relulstm_40/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/Relu_1Р
lstm_40/lstm_cell_40/mul_6Mul"lstm_40/lstm_cell_40/Sigmoid_2:y:0)lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_40/lstm_cell_40/mul_6
%lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_40/TensorArrayV2_1/element_shapeи
lstm_40/TensorArrayV2_1TensorListReserve.lstm_40/TensorArrayV2_1/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_40/TensorArrayV2_1^
lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/time
 lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_40/while/maximum_iterationsz
lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_40/while/loop_counterћ
lstm_40/whileWhile#lstm_40/while/loop_counter:output:0)lstm_40/while/maximum_iterations:output:0lstm_40/time:output:0 lstm_40/TensorArrayV2_1:handle:0lstm_40/zeros:output:0lstm_40/zeros_1:output:0 lstm_40/strided_slice_1:output:0?lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_40_lstm_cell_40_split_readvariableop_resource4lstm_40_lstm_cell_40_split_1_readvariableop_resource,lstm_40_lstm_cell_40_readvariableop_resource*
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
lstm_40_while_body_1384006*&
condR
lstm_40_while_cond_1384005*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_40/whileХ
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_40/TensorArrayV2Stack/TensorListStackTensorListStacklstm_40/while:output:3Alstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_40/TensorArrayV2Stack/TensorListStack
lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_40/strided_slice_3/stack
lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_40/strided_slice_3/stack_1
lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_40/strided_slice_3/stack_2Ъ
lstm_40/strided_slice_3StridedSlice3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_40/strided_slice_3/stack:output:0(lstm_40/strided_slice_3/stack_1:output:0(lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_40/strided_slice_3
lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_40/transpose_1/permХ
lstm_40/transpose_1	Transpose3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_40/transpose_1v
lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_40/runtimeЈ
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_48/MatMul/ReadVariableOpЈ
dense_48/MatMulMatMul lstm_40/strided_slice_3:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/MatMulЇ
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_48/BiasAdd/ReadVariableOpЅ
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/BiasAdds
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_48/ReluЈ
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_49/MatMul/ReadVariableOpЃ
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_49/MatMulЇ
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOpЅ
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_49/BiasAddm
reshape_24/ShapeShapedense_49/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_24/Shape
reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_24/strided_slice/stack
 reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_1
 reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_2Є
reshape_24/strided_sliceStridedSlicereshape_24/Shape:output:0'reshape_24/strided_slice/stack:output:0)reshape_24/strided_slice/stack_1:output:0)reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_24/strided_slicez
reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/1z
reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/2з
reshape_24/Reshape/shapePack!reshape_24/strided_slice:output:0#reshape_24/Reshape/shape/1:output:0#reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_24/Reshape/shapeЇ
reshape_24/ReshapeReshapedense_49/BiasAdd:output:0!reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_24/Reshapeђ
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_40_lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/mulЧ
/dense_49/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_49/bias/Regularizer/Square/ReadVariableOpЌ
 dense_49/bias/Regularizer/SquareSquare7dense_49/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_49/bias/Regularizer/Square
dense_49/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_49/bias/Regularizer/ConstЖ
dense_49/bias/Regularizer/SumSum$dense_49/bias/Regularizer/Square:y:0(dense_49/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/Sum
dense_49/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_49/bias/Regularizer/mul/xИ
dense_49/bias/Regularizer/mulMul(dense_49/bias/Regularizer/mul/x:output:0&dense_49/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_49/bias/Regularizer/mulz
IdentityIdentityreshape_24/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp0^dense_49/bias/Regularizer/Square/ReadVariableOp$^lstm_40/lstm_cell_40/ReadVariableOp&^lstm_40/lstm_cell_40/ReadVariableOp_1&^lstm_40/lstm_cell_40/ReadVariableOp_2&^lstm_40/lstm_cell_40/ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp*^lstm_40/lstm_cell_40/split/ReadVariableOp,^lstm_40/lstm_cell_40/split_1/ReadVariableOp^lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2b
/dense_49/bias/Regularizer/Square/ReadVariableOp/dense_49/bias/Regularizer/Square/ReadVariableOp2J
#lstm_40/lstm_cell_40/ReadVariableOp#lstm_40/lstm_cell_40/ReadVariableOp2N
%lstm_40/lstm_cell_40/ReadVariableOp_1%lstm_40/lstm_cell_40/ReadVariableOp_12N
%lstm_40/lstm_cell_40/ReadVariableOp_2%lstm_40/lstm_cell_40/ReadVariableOp_22N
%lstm_40/lstm_cell_40/ReadVariableOp_3%lstm_40/lstm_cell_40/ReadVariableOp_32~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_40/lstm_cell_40/split/ReadVariableOp)lstm_40/lstm_cell_40/split/ReadVariableOp2Z
+lstm_40/lstm_cell_40/split_1/ReadVariableOp+lstm_40/lstm_cell_40/split_1/ReadVariableOp2
lstm_40/whilelstm_40/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ	
Њ
/__inference_sequential_16_layer_call_fn_1383305
input_17
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_13832882
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
input_17
Ы

ш
lstm_40_while_cond_1384308,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3.
*lstm_40_while_less_lstm_40_strided_slice_1E
Alstm_40_while_lstm_40_while_cond_1384308___redundant_placeholder0E
Alstm_40_while_lstm_40_while_cond_1384308___redundant_placeholder1E
Alstm_40_while_lstm_40_while_cond_1384308___redundant_placeholder2E
Alstm_40_while_lstm_40_while_cond_1384308___redundant_placeholder3
lstm_40_while_identity

lstm_40/while/LessLesslstm_40_while_placeholder*lstm_40_while_less_lstm_40_strided_slice_1*
T0*
_output_shapes
: 2
lstm_40/while/Lessu
lstm_40/while/IdentityIdentitylstm_40/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_40/while/Identity"9
lstm_40_while_identitylstm_40/while/Identity:output:0*(
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
)__inference_lstm_40_layer_call_fn_1384541

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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13832132
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
Р
И
)__inference_lstm_40_layer_call_fn_1384519
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_13823932
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
ЯR
ъ
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1382304

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2,
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
R
Щ
D__inference_lstm_40_layer_call_and_return_conditional_losses_1382690

inputs'
lstm_cell_40_1382602:	#
lstm_cell_40_1382604:	'
lstm_cell_40_1382606:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_40/StatefulPartitionedCallЂwhileD
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
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_40_1382602lstm_cell_40_1382604lstm_cell_40_1382606*
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_13825372&
$lstm_cell_40/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_40_1382602lstm_cell_40_1382604lstm_cell_40_1382606*
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
while_body_1382615*
condR
while_cond_1382614*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_40_1382602*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_40/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
кЯ
Ј
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385652

inputs=
*lstm_cell_40_split_readvariableop_resource:	;
,lstm_cell_40_split_1_readvariableop_resource:	7
$lstm_cell_40_readvariableop_resource:	 
identityЂ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_40/ReadVariableOpЂlstm_cell_40/ReadVariableOp_1Ђlstm_cell_40/ReadVariableOp_2Ђlstm_cell_40/ReadVariableOp_3Ђ!lstm_cell_40/split/ReadVariableOpЂ#lstm_cell_40/split_1/ReadVariableOpЂwhileD
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
lstm_cell_40/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_40/ones_like/Shape
lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_40/ones_like/ConstИ
lstm_cell_40/ones_likeFill%lstm_cell_40/ones_like/Shape:output:0%lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/ones_like}
lstm_cell_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout/ConstГ
lstm_cell_40/dropout/MulMullstm_cell_40/ones_like:output:0#lstm_cell_40/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul
lstm_cell_40/dropout/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout/Shapeј
1lstm_cell_40/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_40/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2сщ23
1lstm_cell_40/dropout/random_uniform/RandomUniform
#lstm_cell_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_40/dropout/GreaterEqual/yђ
!lstm_cell_40/dropout/GreaterEqualGreaterEqual:lstm_cell_40/dropout/random_uniform/RandomUniform:output:0,lstm_cell_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_40/dropout/GreaterEqualІ
lstm_cell_40/dropout/CastCast%lstm_cell_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/CastЎ
lstm_cell_40/dropout/Mul_1Mullstm_cell_40/dropout/Mul:z:0lstm_cell_40/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout/Mul_1
lstm_cell_40/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_1/ConstЙ
lstm_cell_40/dropout_1/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul
lstm_cell_40/dropout_1/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_1/Shape§
3lstm_cell_40/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЇС$25
3lstm_cell_40/dropout_1/random_uniform/RandomUniform
%lstm_cell_40/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_1/GreaterEqual/yњ
#lstm_cell_40/dropout_1/GreaterEqualGreaterEqual<lstm_cell_40/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_1/GreaterEqualЌ
lstm_cell_40/dropout_1/CastCast'lstm_cell_40/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/CastЖ
lstm_cell_40/dropout_1/Mul_1Mullstm_cell_40/dropout_1/Mul:z:0lstm_cell_40/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_1/Mul_1
lstm_cell_40/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_2/ConstЙ
lstm_cell_40/dropout_2/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul
lstm_cell_40/dropout_2/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_2/Shapeў
3lstm_cell_40/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ђе25
3lstm_cell_40/dropout_2/random_uniform/RandomUniform
%lstm_cell_40/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_2/GreaterEqual/yњ
#lstm_cell_40/dropout_2/GreaterEqualGreaterEqual<lstm_cell_40/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_2/GreaterEqualЌ
lstm_cell_40/dropout_2/CastCast'lstm_cell_40/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/CastЖ
lstm_cell_40/dropout_2/Mul_1Mullstm_cell_40/dropout_2/Mul:z:0lstm_cell_40/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_2/Mul_1
lstm_cell_40/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_40/dropout_3/ConstЙ
lstm_cell_40/dropout_3/MulMullstm_cell_40/ones_like:output:0%lstm_cell_40/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul
lstm_cell_40/dropout_3/ShapeShapelstm_cell_40/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_40/dropout_3/Shape§
3lstm_cell_40/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_40/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ь~25
3lstm_cell_40/dropout_3/random_uniform/RandomUniform
%lstm_cell_40/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_40/dropout_3/GreaterEqual/yњ
#lstm_cell_40/dropout_3/GreaterEqualGreaterEqual<lstm_cell_40/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_40/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_40/dropout_3/GreaterEqualЌ
lstm_cell_40/dropout_3/CastCast'lstm_cell_40/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/CastЖ
lstm_cell_40/dropout_3/Mul_1Mullstm_cell_40/dropout_3/Mul:z:0lstm_cell_40/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/dropout_3/Mul_1~
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_40/split/split_dimВ
!lstm_cell_40/split/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_40/split/ReadVariableOpл
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0)lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_40/split
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMulЁ
lstm_cell_40/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_1Ё
lstm_cell_40/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_2Ё
lstm_cell_40/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_3
lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_40/split_1/split_dimД
#lstm_cell_40/split_1/ReadVariableOpReadVariableOp,lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_40/split_1/ReadVariableOpг
lstm_cell_40/split_1Split'lstm_cell_40/split_1/split_dim:output:0+lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_40/split_1Ї
lstm_cell_40/BiasAddBiasAddlstm_cell_40/MatMul:product:0lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd­
lstm_cell_40/BiasAdd_1BiasAddlstm_cell_40/MatMul_1:product:0lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_1­
lstm_cell_40/BiasAdd_2BiasAddlstm_cell_40/MatMul_2:product:0lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_2­
lstm_cell_40/BiasAdd_3BiasAddlstm_cell_40/MatMul_3:product:0lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/BiasAdd_3
lstm_cell_40/mulMulzeros:output:0lstm_cell_40/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul
lstm_cell_40/mul_1Mulzeros:output:0 lstm_cell_40/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_1
lstm_cell_40/mul_2Mulzeros:output:0 lstm_cell_40/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_2
lstm_cell_40/mul_3Mulzeros:output:0 lstm_cell_40/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_3 
lstm_cell_40/ReadVariableOpReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp
 lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_40/strided_slice/stack
"lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice/stack_1
"lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_40/strided_slice/stack_2Ъ
lstm_cell_40/strided_sliceStridedSlice#lstm_cell_40/ReadVariableOp:value:0)lstm_cell_40/strided_slice/stack:output:0+lstm_cell_40/strided_slice/stack_1:output:0+lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_sliceЅ
lstm_cell_40/MatMul_4MatMullstm_cell_40/mul:z:0#lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_4
lstm_cell_40/addAddV2lstm_cell_40/BiasAdd:output:0lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add
lstm_cell_40/SigmoidSigmoidlstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/SigmoidЄ
lstm_cell_40/ReadVariableOp_1ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_1
"lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_40/strided_slice_1/stack
$lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_40/strided_slice_1/stack_1
$lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_1/stack_2ж
lstm_cell_40/strided_slice_1StridedSlice%lstm_cell_40/ReadVariableOp_1:value:0+lstm_cell_40/strided_slice_1/stack:output:0-lstm_cell_40/strided_slice_1/stack_1:output:0-lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_1Љ
lstm_cell_40/MatMul_5MatMullstm_cell_40/mul_1:z:0%lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_5Ѕ
lstm_cell_40/add_1AddV2lstm_cell_40/BiasAdd_1:output:0lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_1
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_1
lstm_cell_40/mul_4Mullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_4Є
lstm_cell_40/ReadVariableOp_2ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_2
"lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_40/strided_slice_2/stack
$lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_40/strided_slice_2/stack_1
$lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_2/stack_2ж
lstm_cell_40/strided_slice_2StridedSlice%lstm_cell_40/ReadVariableOp_2:value:0+lstm_cell_40/strided_slice_2/stack:output:0-lstm_cell_40/strided_slice_2/stack_1:output:0-lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_2Љ
lstm_cell_40/MatMul_6MatMullstm_cell_40/mul_2:z:0%lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_6Ѕ
lstm_cell_40/add_2AddV2lstm_cell_40/BiasAdd_2:output:0lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_2x
lstm_cell_40/ReluRelulstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu
lstm_cell_40/mul_5Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_5
lstm_cell_40/add_3AddV2lstm_cell_40/mul_4:z:0lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_3Є
lstm_cell_40/ReadVariableOp_3ReadVariableOp$lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_40/ReadVariableOp_3
"lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_40/strided_slice_3/stack
$lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_40/strided_slice_3/stack_1
$lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_40/strided_slice_3/stack_2ж
lstm_cell_40/strided_slice_3StridedSlice%lstm_cell_40/ReadVariableOp_3:value:0+lstm_cell_40/strided_slice_3/stack:output:0-lstm_cell_40/strided_slice_3/stack_1:output:0-lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_40/strided_slice_3Љ
lstm_cell_40/MatMul_7MatMullstm_cell_40/mul_3:z:0%lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/MatMul_7Ѕ
lstm_cell_40/add_4AddV2lstm_cell_40/BiasAdd_3:output:0lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/add_4
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Sigmoid_2|
lstm_cell_40/Relu_1Relulstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/Relu_1 
lstm_cell_40/mul_6Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_40/mul_6
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_40_split_readvariableop_resource,lstm_cell_40_split_1_readvariableop_resource$lstm_cell_40_readvariableop_resource*
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
while_body_1385487*
condR
while_cond_1385486*K
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_40/ReadVariableOp^lstm_cell_40/ReadVariableOp_1^lstm_cell_40/ReadVariableOp_2^lstm_cell_40/ReadVariableOp_3"^lstm_cell_40/split/ReadVariableOp$^lstm_cell_40/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_40/ReadVariableOplstm_cell_40/ReadVariableOp2>
lstm_cell_40/ReadVariableOp_1lstm_cell_40/ReadVariableOp_12>
lstm_cell_40/ReadVariableOp_2lstm_cell_40/ReadVariableOp_22>
lstm_cell_40/ReadVariableOp_3lstm_cell_40/ReadVariableOp_32F
!lstm_cell_40/split/ReadVariableOp!lstm_cell_40/split/ReadVariableOp2J
#lstm_cell_40/split_1/ReadVariableOp#lstm_cell_40/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Аў
	
"__inference__wrapped_model_1382180
input_17S
@sequential_16_lstm_40_lstm_cell_40_split_readvariableop_resource:	Q
Bsequential_16_lstm_40_lstm_cell_40_split_1_readvariableop_resource:	M
:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource:	 G
5sequential_16_dense_48_matmul_readvariableop_resource:  D
6sequential_16_dense_48_biasadd_readvariableop_resource: G
5sequential_16_dense_49_matmul_readvariableop_resource: D
6sequential_16_dense_49_biasadd_readvariableop_resource:
identityЂ-sequential_16/dense_48/BiasAdd/ReadVariableOpЂ,sequential_16/dense_48/MatMul/ReadVariableOpЂ-sequential_16/dense_49/BiasAdd/ReadVariableOpЂ,sequential_16/dense_49/MatMul/ReadVariableOpЂ1sequential_16/lstm_40/lstm_cell_40/ReadVariableOpЂ3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_1Ђ3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_2Ђ3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_3Ђ7sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOpЂ9sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOpЂsequential_16/lstm_40/whiler
sequential_16/lstm_40/ShapeShapeinput_17*
T0*
_output_shapes
:2
sequential_16/lstm_40/Shape 
)sequential_16/lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_16/lstm_40/strided_slice/stackЄ
+sequential_16/lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_16/lstm_40/strided_slice/stack_1Є
+sequential_16/lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_16/lstm_40/strided_slice/stack_2ц
#sequential_16/lstm_40/strided_sliceStridedSlice$sequential_16/lstm_40/Shape:output:02sequential_16/lstm_40/strided_slice/stack:output:04sequential_16/lstm_40/strided_slice/stack_1:output:04sequential_16/lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_16/lstm_40/strided_slice
!sequential_16/lstm_40/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_16/lstm_40/zeros/mul/yФ
sequential_16/lstm_40/zeros/mulMul,sequential_16/lstm_40/strided_slice:output:0*sequential_16/lstm_40/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_16/lstm_40/zeros/mul
"sequential_16/lstm_40/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_16/lstm_40/zeros/Less/yП
 sequential_16/lstm_40/zeros/LessLess#sequential_16/lstm_40/zeros/mul:z:0+sequential_16/lstm_40/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_16/lstm_40/zeros/Less
$sequential_16/lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_16/lstm_40/zeros/packed/1л
"sequential_16/lstm_40/zeros/packedPack,sequential_16/lstm_40/strided_slice:output:0-sequential_16/lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_16/lstm_40/zeros/packed
!sequential_16/lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_16/lstm_40/zeros/ConstЭ
sequential_16/lstm_40/zerosFill+sequential_16/lstm_40/zeros/packed:output:0*sequential_16/lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_16/lstm_40/zeros
#sequential_16/lstm_40/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_16/lstm_40/zeros_1/mul/yЪ
!sequential_16/lstm_40/zeros_1/mulMul,sequential_16/lstm_40/strided_slice:output:0,sequential_16/lstm_40/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_16/lstm_40/zeros_1/mul
$sequential_16/lstm_40/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_16/lstm_40/zeros_1/Less/yЧ
"sequential_16/lstm_40/zeros_1/LessLess%sequential_16/lstm_40/zeros_1/mul:z:0-sequential_16/lstm_40/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_16/lstm_40/zeros_1/Less
&sequential_16/lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_16/lstm_40/zeros_1/packed/1с
$sequential_16/lstm_40/zeros_1/packedPack,sequential_16/lstm_40/strided_slice:output:0/sequential_16/lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_16/lstm_40/zeros_1/packed
#sequential_16/lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_16/lstm_40/zeros_1/Constе
sequential_16/lstm_40/zeros_1Fill-sequential_16/lstm_40/zeros_1/packed:output:0,sequential_16/lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_16/lstm_40/zeros_1Ё
$sequential_16/lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_16/lstm_40/transpose/permО
sequential_16/lstm_40/transpose	Transposeinput_17-sequential_16/lstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_16/lstm_40/transpose
sequential_16/lstm_40/Shape_1Shape#sequential_16/lstm_40/transpose:y:0*
T0*
_output_shapes
:2
sequential_16/lstm_40/Shape_1Є
+sequential_16/lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_16/lstm_40/strided_slice_1/stackЈ
-sequential_16/lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_16/lstm_40/strided_slice_1/stack_1Ј
-sequential_16/lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_16/lstm_40/strided_slice_1/stack_2ђ
%sequential_16/lstm_40/strided_slice_1StridedSlice&sequential_16/lstm_40/Shape_1:output:04sequential_16/lstm_40/strided_slice_1/stack:output:06sequential_16/lstm_40/strided_slice_1/stack_1:output:06sequential_16/lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_16/lstm_40/strided_slice_1Б
1sequential_16/lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ23
1sequential_16/lstm_40/TensorArrayV2/element_shape
#sequential_16/lstm_40/TensorArrayV2TensorListReserve:sequential_16/lstm_40/TensorArrayV2/element_shape:output:0.sequential_16/lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_16/lstm_40/TensorArrayV2ы
Ksequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2M
Ksequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeа
=sequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_16/lstm_40/transpose:y:0Tsequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensorЄ
+sequential_16/lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_16/lstm_40/strided_slice_2/stackЈ
-sequential_16/lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_16/lstm_40/strided_slice_2/stack_1Ј
-sequential_16/lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_16/lstm_40/strided_slice_2/stack_2
%sequential_16/lstm_40/strided_slice_2StridedSlice#sequential_16/lstm_40/transpose:y:04sequential_16/lstm_40/strided_slice_2/stack:output:06sequential_16/lstm_40/strided_slice_2/stack_1:output:06sequential_16/lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2'
%sequential_16/lstm_40/strided_slice_2М
2sequential_16/lstm_40/lstm_cell_40/ones_like/ShapeShape$sequential_16/lstm_40/zeros:output:0*
T0*
_output_shapes
:24
2sequential_16/lstm_40/lstm_cell_40/ones_like/Shape­
2sequential_16/lstm_40/lstm_cell_40/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_16/lstm_40/lstm_cell_40/ones_like/Const
,sequential_16/lstm_40/lstm_cell_40/ones_likeFill;sequential_16/lstm_40/lstm_cell_40/ones_like/Shape:output:0;sequential_16/lstm_40/lstm_cell_40/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/ones_likeЊ
2sequential_16/lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_16/lstm_40/lstm_cell_40/split/split_dimє
7sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOpReadVariableOp@sequential_16_lstm_40_lstm_cell_40_split_readvariableop_resource*
_output_shapes
:	*
dtype029
7sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOpГ
(sequential_16/lstm_40/lstm_cell_40/splitSplit;sequential_16/lstm_40/lstm_cell_40/split/split_dim:output:0?sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2*
(sequential_16/lstm_40/lstm_cell_40/splitѕ
)sequential_16/lstm_40/lstm_cell_40/MatMulMatMul.sequential_16/lstm_40/strided_slice_2:output:01sequential_16/lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_16/lstm_40/lstm_cell_40/MatMulљ
+sequential_16/lstm_40/lstm_cell_40/MatMul_1MatMul.sequential_16/lstm_40/strided_slice_2:output:01sequential_16/lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_1љ
+sequential_16/lstm_40/lstm_cell_40/MatMul_2MatMul.sequential_16/lstm_40/strided_slice_2:output:01sequential_16/lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_2љ
+sequential_16/lstm_40/lstm_cell_40/MatMul_3MatMul.sequential_16/lstm_40/strided_slice_2:output:01sequential_16/lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_3Ў
4sequential_16/lstm_40/lstm_cell_40/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_16/lstm_40/lstm_cell_40/split_1/split_dimі
9sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOpReadVariableOpBsequential_16_lstm_40_lstm_cell_40_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOpЋ
*sequential_16/lstm_40/lstm_cell_40/split_1Split=sequential_16/lstm_40/lstm_cell_40/split_1/split_dim:output:0Asequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2,
*sequential_16/lstm_40/lstm_cell_40/split_1џ
*sequential_16/lstm_40/lstm_cell_40/BiasAddBiasAdd3sequential_16/lstm_40/lstm_cell_40/MatMul:product:03sequential_16/lstm_40/lstm_cell_40/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_16/lstm_40/lstm_cell_40/BiasAdd
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_1BiasAdd5sequential_16/lstm_40/lstm_cell_40/MatMul_1:product:03sequential_16/lstm_40/lstm_cell_40/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_1
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_2BiasAdd5sequential_16/lstm_40/lstm_cell_40/MatMul_2:product:03sequential_16/lstm_40/lstm_cell_40/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_2
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_3BiasAdd5sequential_16/lstm_40/lstm_cell_40/MatMul_3:product:03sequential_16/lstm_40/lstm_cell_40/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/BiasAdd_3ц
&sequential_16/lstm_40/lstm_cell_40/mulMul$sequential_16/lstm_40/zeros:output:05sequential_16/lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_16/lstm_40/lstm_cell_40/mulъ
(sequential_16/lstm_40/lstm_cell_40/mul_1Mul$sequential_16/lstm_40/zeros:output:05sequential_16/lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_1ъ
(sequential_16/lstm_40/lstm_cell_40/mul_2Mul$sequential_16/lstm_40/zeros:output:05sequential_16/lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_2ъ
(sequential_16/lstm_40/lstm_cell_40/mul_3Mul$sequential_16/lstm_40/zeros:output:05sequential_16/lstm_40/lstm_cell_40/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_3т
1sequential_16/lstm_40/lstm_cell_40/ReadVariableOpReadVariableOp:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype023
1sequential_16/lstm_40/lstm_cell_40/ReadVariableOpС
6sequential_16/lstm_40/lstm_cell_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_16/lstm_40/lstm_cell_40/strided_slice/stackХ
8sequential_16/lstm_40/lstm_cell_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_16/lstm_40/lstm_cell_40/strided_slice/stack_1Х
8sequential_16/lstm_40/lstm_cell_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_16/lstm_40/lstm_cell_40/strided_slice/stack_2Ю
0sequential_16/lstm_40/lstm_cell_40/strided_sliceStridedSlice9sequential_16/lstm_40/lstm_cell_40/ReadVariableOp:value:0?sequential_16/lstm_40/lstm_cell_40/strided_slice/stack:output:0Asequential_16/lstm_40/lstm_cell_40/strided_slice/stack_1:output:0Asequential_16/lstm_40/lstm_cell_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_16/lstm_40/lstm_cell_40/strided_slice§
+sequential_16/lstm_40/lstm_cell_40/MatMul_4MatMul*sequential_16/lstm_40/lstm_cell_40/mul:z:09sequential_16/lstm_40/lstm_cell_40/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_4ї
&sequential_16/lstm_40/lstm_cell_40/addAddV23sequential_16/lstm_40/lstm_cell_40/BiasAdd:output:05sequential_16/lstm_40/lstm_cell_40/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_16/lstm_40/lstm_cell_40/addС
*sequential_16/lstm_40/lstm_cell_40/SigmoidSigmoid*sequential_16/lstm_40/lstm_cell_40/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_16/lstm_40/lstm_cell_40/Sigmoidц
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_1ReadVariableOp:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_1Х
8sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stackЩ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_1Щ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_2к
2sequential_16/lstm_40/lstm_cell_40/strided_slice_1StridedSlice;sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_1:value:0Asequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_1:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_16/lstm_40/lstm_cell_40/strided_slice_1
+sequential_16/lstm_40/lstm_cell_40/MatMul_5MatMul,sequential_16/lstm_40/lstm_cell_40/mul_1:z:0;sequential_16/lstm_40/lstm_cell_40/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_5§
(sequential_16/lstm_40/lstm_cell_40/add_1AddV25sequential_16/lstm_40/lstm_cell_40/BiasAdd_1:output:05sequential_16/lstm_40/lstm_cell_40/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/add_1Ч
,sequential_16/lstm_40/lstm_cell_40/Sigmoid_1Sigmoid,sequential_16/lstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/Sigmoid_1ч
(sequential_16/lstm_40/lstm_cell_40/mul_4Mul0sequential_16/lstm_40/lstm_cell_40/Sigmoid_1:y:0&sequential_16/lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_4ц
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_2ReadVariableOp:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_2Х
8sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stackЩ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_1Щ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_2к
2sequential_16/lstm_40/lstm_cell_40/strided_slice_2StridedSlice;sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_2:value:0Asequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_1:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_16/lstm_40/lstm_cell_40/strided_slice_2
+sequential_16/lstm_40/lstm_cell_40/MatMul_6MatMul,sequential_16/lstm_40/lstm_cell_40/mul_2:z:0;sequential_16/lstm_40/lstm_cell_40/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_6§
(sequential_16/lstm_40/lstm_cell_40/add_2AddV25sequential_16/lstm_40/lstm_cell_40/BiasAdd_2:output:05sequential_16/lstm_40/lstm_cell_40/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/add_2К
'sequential_16/lstm_40/lstm_cell_40/ReluRelu,sequential_16/lstm_40/lstm_cell_40/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_16/lstm_40/lstm_cell_40/Reluє
(sequential_16/lstm_40/lstm_cell_40/mul_5Mul.sequential_16/lstm_40/lstm_cell_40/Sigmoid:y:05sequential_16/lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_5ы
(sequential_16/lstm_40/lstm_cell_40/add_3AddV2,sequential_16/lstm_40/lstm_cell_40/mul_4:z:0,sequential_16/lstm_40/lstm_cell_40/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/add_3ц
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_3ReadVariableOp:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_3Х
8sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2:
8sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stackЩ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_1Щ
:sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_2к
2sequential_16/lstm_40/lstm_cell_40/strided_slice_3StridedSlice;sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_3:value:0Asequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_1:output:0Csequential_16/lstm_40/lstm_cell_40/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_16/lstm_40/lstm_cell_40/strided_slice_3
+sequential_16/lstm_40/lstm_cell_40/MatMul_7MatMul,sequential_16/lstm_40/lstm_cell_40/mul_3:z:0;sequential_16/lstm_40/lstm_cell_40/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_16/lstm_40/lstm_cell_40/MatMul_7§
(sequential_16/lstm_40/lstm_cell_40/add_4AddV25sequential_16/lstm_40/lstm_cell_40/BiasAdd_3:output:05sequential_16/lstm_40/lstm_cell_40/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/add_4Ч
,sequential_16/lstm_40/lstm_cell_40/Sigmoid_2Sigmoid,sequential_16/lstm_40/lstm_cell_40/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_16/lstm_40/lstm_cell_40/Sigmoid_2О
)sequential_16/lstm_40/lstm_cell_40/Relu_1Relu,sequential_16/lstm_40/lstm_cell_40/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_16/lstm_40/lstm_cell_40/Relu_1ј
(sequential_16/lstm_40/lstm_cell_40/mul_6Mul0sequential_16/lstm_40/lstm_cell_40/Sigmoid_2:y:07sequential_16/lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_16/lstm_40/lstm_cell_40/mul_6Л
3sequential_16/lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    25
3sequential_16/lstm_40/TensorArrayV2_1/element_shape
%sequential_16/lstm_40/TensorArrayV2_1TensorListReserve<sequential_16/lstm_40/TensorArrayV2_1/element_shape:output:0.sequential_16/lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_16/lstm_40/TensorArrayV2_1z
sequential_16/lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_16/lstm_40/timeЋ
.sequential_16/lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.sequential_16/lstm_40/while/maximum_iterations
(sequential_16/lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_16/lstm_40/while/loop_counterЭ
sequential_16/lstm_40/whileWhile1sequential_16/lstm_40/while/loop_counter:output:07sequential_16/lstm_40/while/maximum_iterations:output:0#sequential_16/lstm_40/time:output:0.sequential_16/lstm_40/TensorArrayV2_1:handle:0$sequential_16/lstm_40/zeros:output:0&sequential_16/lstm_40/zeros_1:output:0.sequential_16/lstm_40/strided_slice_1:output:0Msequential_16/lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_16_lstm_40_lstm_cell_40_split_readvariableop_resourceBsequential_16_lstm_40_lstm_cell_40_split_1_readvariableop_resource:sequential_16_lstm_40_lstm_cell_40_readvariableop_resource*
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
(sequential_16_lstm_40_while_body_1382031*4
cond,R*
(sequential_16_lstm_40_while_cond_1382030*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_16/lstm_40/whileс
Fsequential_16/lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2H
Fsequential_16/lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeР
8sequential_16/lstm_40/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_16/lstm_40/while:output:3Osequential_16/lstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02:
8sequential_16/lstm_40/TensorArrayV2Stack/TensorListStack­
+sequential_16/lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2-
+sequential_16/lstm_40/strided_slice_3/stackЈ
-sequential_16/lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_16/lstm_40/strided_slice_3/stack_1Ј
-sequential_16/lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_16/lstm_40/strided_slice_3/stack_2
%sequential_16/lstm_40/strided_slice_3StridedSliceAsequential_16/lstm_40/TensorArrayV2Stack/TensorListStack:tensor:04sequential_16/lstm_40/strided_slice_3/stack:output:06sequential_16/lstm_40/strided_slice_3/stack_1:output:06sequential_16/lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2'
%sequential_16/lstm_40/strided_slice_3Ѕ
&sequential_16/lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_16/lstm_40/transpose_1/perm§
!sequential_16/lstm_40/transpose_1	TransposeAsequential_16/lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_16/lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2#
!sequential_16/lstm_40/transpose_1
sequential_16/lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_16/lstm_40/runtimeв
,sequential_16/dense_48/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_48_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02.
,sequential_16/dense_48/MatMul/ReadVariableOpр
sequential_16/dense_48/MatMulMatMul.sequential_16/lstm_40/strided_slice_3:output:04sequential_16/dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_16/dense_48/MatMulб
-sequential_16/dense_48/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_16/dense_48/BiasAdd/ReadVariableOpн
sequential_16/dense_48/BiasAddBiasAdd'sequential_16/dense_48/MatMul:product:05sequential_16/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
sequential_16/dense_48/BiasAdd
sequential_16/dense_48/ReluRelu'sequential_16/dense_48/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_16/dense_48/Reluв
,sequential_16/dense_49/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_49_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_16/dense_49/MatMul/ReadVariableOpл
sequential_16/dense_49/MatMulMatMul)sequential_16/dense_48/Relu:activations:04sequential_16/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_16/dense_49/MatMulб
-sequential_16/dense_49/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_49/BiasAdd/ReadVariableOpн
sequential_16/dense_49/BiasAddBiasAdd'sequential_16/dense_49/MatMul:product:05sequential_16/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_16/dense_49/BiasAdd
sequential_16/reshape_24/ShapeShape'sequential_16/dense_49/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_16/reshape_24/ShapeІ
,sequential_16/reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_16/reshape_24/strided_slice/stackЊ
.sequential_16/reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_16/reshape_24/strided_slice/stack_1Њ
.sequential_16/reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_16/reshape_24/strided_slice/stack_2ј
&sequential_16/reshape_24/strided_sliceStridedSlice'sequential_16/reshape_24/Shape:output:05sequential_16/reshape_24/strided_slice/stack:output:07sequential_16/reshape_24/strided_slice/stack_1:output:07sequential_16/reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_16/reshape_24/strided_slice
(sequential_16/reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_16/reshape_24/Reshape/shape/1
(sequential_16/reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_16/reshape_24/Reshape/shape/2
&sequential_16/reshape_24/Reshape/shapePack/sequential_16/reshape_24/strided_slice:output:01sequential_16/reshape_24/Reshape/shape/1:output:01sequential_16/reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/reshape_24/Reshape/shapeп
 sequential_16/reshape_24/ReshapeReshape'sequential_16/dense_49/BiasAdd:output:0/sequential_16/reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_16/reshape_24/Reshape
IdentityIdentity)sequential_16/reshape_24/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityі
NoOpNoOp.^sequential_16/dense_48/BiasAdd/ReadVariableOp-^sequential_16/dense_48/MatMul/ReadVariableOp.^sequential_16/dense_49/BiasAdd/ReadVariableOp-^sequential_16/dense_49/MatMul/ReadVariableOp2^sequential_16/lstm_40/lstm_cell_40/ReadVariableOp4^sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_14^sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_24^sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_38^sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOp:^sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOp^sequential_16/lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2^
-sequential_16/dense_48/BiasAdd/ReadVariableOp-sequential_16/dense_48/BiasAdd/ReadVariableOp2\
,sequential_16/dense_48/MatMul/ReadVariableOp,sequential_16/dense_48/MatMul/ReadVariableOp2^
-sequential_16/dense_49/BiasAdd/ReadVariableOp-sequential_16/dense_49/BiasAdd/ReadVariableOp2\
,sequential_16/dense_49/MatMul/ReadVariableOp,sequential_16/dense_49/MatMul/ReadVariableOp2f
1sequential_16/lstm_40/lstm_cell_40/ReadVariableOp1sequential_16/lstm_40/lstm_cell_40/ReadVariableOp2j
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_13sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_12j
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_23sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_22j
3sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_33sequential_16/lstm_40/lstm_cell_40/ReadVariableOp_32r
7sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOp7sequential_16/lstm_40/lstm_cell_40/split/ReadVariableOp2v
9sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOp9sequential_16/lstm_40/lstm_cell_40/split_1/ReadVariableOp2:
sequential_16/lstm_40/whilesequential_16/lstm_40/while:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_17

і
E__inference_dense_48_layer_call_and_return_conditional_losses_1385663

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
к
Ш
while_cond_1382614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1382614___redundant_placeholder05
1while_while_cond_1382614___redundant_placeholder15
1while_while_cond_1382614___redundant_placeholder25
1while_while_cond_1382614___redundant_placeholder3
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
Їv
ъ
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1382537

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
dropout/Shapeб
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2чОл2&
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
seed2Т"2(
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
seed2м2(
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
seed2ЭЃ2(
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOpл
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareSquareElstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_40/lstm_cell_40/kernel/Regularizer/SquareЏ
-lstm_40/lstm_cell_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_40/lstm_cell_40/kernel/Regularizer/Constю
+lstm_40/lstm_cell_40/kernel/Regularizer/SumSum2lstm_40/lstm_cell_40/kernel/Regularizer/Square:y:06lstm_40/lstm_cell_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/SumЃ
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_40/lstm_cell_40/kernel/Regularizer/mul/x№
+lstm_40/lstm_cell_40/kernel/Regularizer/mulMul6lstm_40/lstm_cell_40/kernel/Regularizer/mul/x:output:04lstm_40/lstm_cell_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_40/lstm_cell_40/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp=lstm_40/lstm_cell_40/kernel/Regularizer/Square/ReadVariableOp2,
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
к
Ш
while_cond_1384936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1384936___redundant_placeholder05
1while_while_cond_1384936___redundant_placeholder15
1while_while_cond_1384936___redundant_placeholder25
1while_while_cond_1384936___redundant_placeholder3
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
ѓ

*__inference_dense_48_layer_call_fn_1385672

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
E__inference_dense_48_layer_call_and_return_conditional_losses_13832322
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

c
G__inference_reshape_24_layer_call_and_return_conditional_losses_1383273

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
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*З
serving_defaultЃ
A
input_175
serving_default_input_17:0џџџџџџџџџB

reshape_244
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
	variables
trainable_variables
regularization_losses
		keras_api


signatures
`__call__
a_default_save_signature
*b&call_and_return_all_conditional_losses"
_tf_keras_sequential
У
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
 	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
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
*layer_regularization_losses
	variables
trainable_variables
+layer_metrics
,metrics
regularization_losses
-non_trainable_variables
`__call__
a_default_save_signature
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
/	variables
0trainable_variables
1regularization_losses
2	keras_api
*m&call_and_return_all_conditional_losses
n__call__"
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
4layer_regularization_losses

5states
	variables
trainable_variables
6layer_metrics
7metrics
regularization_losses
8non_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_48/kernel
: 2dense_48/bias
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
	variables
:non_trainable_variables
trainable_variables
;layer_metrics
<metrics
regularization_losses
=layer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_49/kernel
:2dense_49/bias
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
	variables
?non_trainable_variables
trainable_variables
@layer_metrics
Ametrics
regularization_losses
Blayer_regularization_losses
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Clayers
	variables
Dnon_trainable_variables
trainable_variables
Elayer_metrics
Fmetrics
regularization_losses
Glayer_regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	2lstm_40/lstm_cell_40/kernel
8:6	 2%lstm_40/lstm_cell_40/recurrent_kernel
(:&2lstm_40/lstm_cell_40/bias
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
H0"
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
/	variables
Jnon_trainable_variables
0trainable_variables
Klayer_metrics
Lmetrics
1regularization_losses
Mlayer_regularization_losses
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
k0"
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
 "
trackable_list_wrapper
'
o0"
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
&:$  2Adam/dense_48/kernel/m
 : 2Adam/dense_48/bias/m
&:$ 2Adam/dense_49/kernel/m
 :2Adam/dense_49/bias/m
3:1	2"Adam/lstm_40/lstm_cell_40/kernel/m
=:;	 2,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m
-:+2 Adam/lstm_40/lstm_cell_40/bias/m
&:$  2Adam/dense_48/kernel/v
 : 2Adam/dense_48/bias/v
&:$ 2Adam/dense_49/kernel/v
 :2Adam/dense_49/bias/v
3:1	2"Adam/lstm_40/lstm_cell_40/kernel/v
=:;	 2,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v
-:+2 Adam/lstm_40/lstm_cell_40/bias/v
2
/__inference_sequential_16_layer_call_fn_1383305
/__inference_sequential_16_layer_call_fn_1383877
/__inference_sequential_16_layer_call_fn_1383896
/__inference_sequential_16_layer_call_fn_1383751Р
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
ЮBЫ
"__inference__wrapped_model_1382180input_17"
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
і2ѓ
J__inference_sequential_16_layer_call_and_return_conditional_losses_1384167
J__inference_sequential_16_layer_call_and_return_conditional_losses_1384502
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383785
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383819Р
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
)__inference_lstm_40_layer_call_fn_1384519
)__inference_lstm_40_layer_call_fn_1384530
)__inference_lstm_40_layer_call_fn_1384541
)__inference_lstm_40_layer_call_fn_1384552е
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1384795
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385102
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385345
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385652е
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
я2ь
E__inference_dense_48_layer_call_and_return_conditional_losses_1385663Ђ
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
*__inference_dense_48_layer_call_fn_1385672Ђ
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
E__inference_dense_49_layer_call_and_return_conditional_losses_1385694Ђ
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
*__inference_dense_49_layer_call_fn_1385703Ђ
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_1385716Ђ
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
,__inference_reshape_24_layer_call_fn_1385721Ђ
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
__inference_loss_fn_0_1385732
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
%__inference_signature_wrapper_1383858input_17"
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
к2з
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385819
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385932О
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
Є2Ё
.__inference_lstm_cell_40_layer_call_fn_1385949
.__inference_lstm_cell_40_layer_call_fn_1385966О
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
__inference_loss_fn_1_1385977
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
"__inference__wrapped_model_1382180}&('5Ђ2
+Ђ(
&#
input_17џџџџџџџџџ
Њ ";Њ8
6

reshape_24(%

reshape_24џџџџџџџџџЅ
E__inference_dense_48_layer_call_and_return_conditional_losses_1385663\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_48_layer_call_fn_1385672O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ѕ
E__inference_dense_49_layer_call_and_return_conditional_losses_1385694\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_49_layer_call_fn_1385703O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ<
__inference_loss_fn_0_1385732Ђ

Ђ 
Њ " <
__inference_loss_fn_1_1385977&Ђ

Ђ 
Њ " Х
D__inference_lstm_40_layer_call_and_return_conditional_losses_1384795}&('OЂL
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385102}&('OЂL
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385345m&('?Ђ<
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1385652m&('?Ђ<
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
)__inference_lstm_40_layer_call_fn_1384519p&('OЂL
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
)__inference_lstm_40_layer_call_fn_1384530p&('OЂL
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
)__inference_lstm_40_layer_call_fn_1384541`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
)__inference_lstm_40_layer_call_fn_1384552`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ Ы
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385819§&('Ђ}
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1385932§&('Ђ}
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
.__inference_lstm_cell_40_layer_call_fn_1385949э&('Ђ}
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
.__inference_lstm_cell_40_layer_call_fn_1385966э&('Ђ}
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
G__inference_reshape_24_layer_call_and_return_conditional_losses_1385716\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
,__inference_reshape_24_layer_call_fn_1385721O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџС
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383785s&('=Ђ:
3Ђ0
&#
input_17џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 С
J__inference_sequential_16_layer_call_and_return_conditional_losses_1383819s&('=Ђ:
3Ђ0
&#
input_17џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 П
J__inference_sequential_16_layer_call_and_return_conditional_losses_1384167q&(';Ђ8
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_1384502q&(';Ђ8
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
/__inference_sequential_16_layer_call_fn_1383305f&('=Ђ:
3Ђ0
&#
input_17џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_16_layer_call_fn_1383751f&('=Ђ:
3Ђ0
&#
input_17џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_16_layer_call_fn_1383877d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_16_layer_call_fn_1383896d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџГ
%__inference_signature_wrapper_1383858&('AЂ>
Ђ 
7Њ4
2
input_17&#
input_17џџџџџџџџџ";Њ8
6

reshape_24(%

reshape_24џџџџџџџџџ