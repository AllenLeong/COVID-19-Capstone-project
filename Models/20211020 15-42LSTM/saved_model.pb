фЦ%
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8лЊ$
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

lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namelstm_8/lstm_cell_8/kernel

-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/kernel*
_output_shapes
:	*
dtype0
Ѓ
#lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *4
shared_name%#lstm_8/lstm_cell_8/recurrent_kernel

7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_8/lstm_cell_8/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_8/lstm_cell_8/bias

+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/bias*
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:  *
dtype0

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

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

: *
dtype0

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

 Adam/lstm_8/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/m

4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/m*
_output_shapes
:	*
dtype0
Б
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
Њ
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

Adam/lstm_8/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_8/lstm_cell_8/bias/m

2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:  *
dtype0

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

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

: *
dtype0

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

 Adam/lstm_8/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/v

4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/v*
_output_shapes
:	*
dtype0
Б
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
Њ
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

Adam/lstm_8/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_8/lstm_cell_8/bias/v

2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Ѓ,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*о+
valueд+Bб+ BЪ+
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
 
1
&0
'1
(2
3
4
5
6
­

)layers
*layer_metrics
trainable_variables
+metrics
,layer_regularization_losses
regularization_losses
-non_trainable_variables
	variables
 

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
Й

3states

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables
	variables
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
­

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
regularization_losses
=non_trainable_variables
	variables
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
­

>layers
?layer_metrics
trainable_variables
@metrics
Alayer_regularization_losses
regularization_losses
Bnon_trainable_variables
	variables
 
 
 
­

Clayers
Dlayer_metrics
trainable_variables
Emetrics
Flayer_regularization_losses
regularization_losses
Gnon_trainable_variables
	variables
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
_]
VARIABLE_VALUElstm_8/lstm_cell_8/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_8/lstm_cell_8/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_8/lstm_cell_8/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

H0
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
­

Ilayers
Jlayer_metrics
/trainable_variables
Kmetrics
Llayer_regularization_losses
0regularization_losses
Mnon_trainable_variables
1	variables
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

VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_5Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
н
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5lstm_8/lstm_cell_8/kernellstm_8/lstm_cell_8/bias#lstm_8/lstm_cell_8/recurrent_kerneldense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
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
GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_162018
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ч
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_8/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpConst*)
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
GPU 2J 8 *(
f#R!
__inference__traced_save_164209
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biastotalcountAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m Adam/lstm_8/lstm_cell_8/kernel/m*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mAdam/lstm_8/lstm_cell_8/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v Adam/lstm_8/lstm_cell_8/kernel/v*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vAdam/lstm_8/lstm_cell_8/bias/v*(
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_164303уЌ#
ъ!
ђ
H__inference_sequential_4_layer_call_and_return_conditional_losses_161472

inputs 
lstm_8_161410:	
lstm_8_161412:	 
lstm_8_161414:	 !
dense_12_161429:  
dense_12_161431: !
dense_13_161445: 
dense_13_161447:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂlstm_8/StatefulPartitionedCallЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_161410lstm_8_161412lstm_8_161414*
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1614092 
lstm_8/StatefulPartitionedCallЕ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_12_161429dense_12_161431*
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
GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1614282"
 dense_12/StatefulPartitionedCallЗ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_161445dense_13_161447*
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
GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1614442"
 dense_13/StatefulPartitionedCallў
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_1614632
reshape_6/PartitionedCallЩ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_8_161410*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/mul
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityѓ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
МА
	
while_body_163041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2!
while/lstm_cell_8/dropout/ConstЧ
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/dropout/Mul
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_8/dropout/Shape
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed228
6while/lstm_cell_8/dropout/random_uniform/RandomUniform
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2*
(while/lstm_cell_8/dropout/GreaterEqual/y
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&while/lstm_cell_8/dropout/GreaterEqualЕ
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_8/dropout/CastТ
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout/Mul_1
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_1/ConstЭ
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_1/Mul
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_1/Shape
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Б2:
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_1/GreaterEqual/y
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_1/GreaterEqualЛ
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_1/CastЪ
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_1/Mul_1
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_2/ConstЭ
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_2/Mul
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_2/Shape
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЫЄ2:
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_2/GreaterEqual/y
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_2/GreaterEqualЛ
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_2/CastЪ
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_2/Mul_1
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_3/ConstЭ
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_3/Mul
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_3/Shape
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2фЛй2:
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_3/GreaterEqual/y
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_3/GreaterEqualЛ
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_3/CastЪ
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_3/Mul_1
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ё
while/lstm_cell_8/mulMulwhile_placeholder_2#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulЇ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1Ї
while/lstm_cell_8/mul_2Mulwhile_placeholder_2%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2Ї
while/lstm_cell_8/mul_3Mulwhile_placeholder_2%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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
аQ
О
B__inference_lstm_8_layer_call_and_return_conditional_losses_160589

inputs%
lstm_cell_8_160501:	!
lstm_cell_8_160503:	%
lstm_cell_8_160505:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂ#lstm_cell_8/StatefulPartitionedCallЂwhileD
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
 :џџџџџџџџџџџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_160501lstm_cell_8_160503lstm_cell_8_160505*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1605002%
#lstm_cell_8/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_160501lstm_cell_8_160503lstm_cell_8_160505*
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
bodyR
while_body_160514*
condR
while_cond_160513*K
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
runtimeЮ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_8_160501*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityК
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
в
г
H__inference_sequential_4_layer_call_and_return_conditional_losses_162283

inputsC
0lstm_8_lstm_cell_8_split_readvariableop_resource:	A
2lstm_8_lstm_cell_8_split_1_readvariableop_resource:	=
*lstm_8_lstm_cell_8_readvariableop_resource:	 9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource: 6
(dense_13_biasadd_readvariableop_resource:
identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂ!lstm_8/lstm_cell_8/ReadVariableOpЂ#lstm_8/lstm_cell_8/ReadVariableOp_1Ђ#lstm_8/lstm_cell_8/ReadVariableOp_2Ђ#lstm_8/lstm_cell_8/ReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂ'lstm_8/lstm_cell_8/split/ReadVariableOpЂ)lstm_8/lstm_cell_8/split_1/ReadVariableOpЂlstm_8/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/Shape
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros/mul/y
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_8/zeros/Less/y
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros/packed/1
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros_1/mul/y
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_8/zeros_1/Less/y
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros_1/packed/1Ѕ
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/zeros_1
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_8/TensorArrayV2/element_shapeЮ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2Э
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2І
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_8/strided_slice_2
"lstm_8/lstm_cell_8/ones_like/ShapeShapelstm_8/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_8/lstm_cell_8/ones_like/Shape
"lstm_8/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm_8/lstm_cell_8/ones_like/Constа
lstm_8/lstm_cell_8/ones_likeFill+lstm_8/lstm_cell_8/ones_like/Shape:output:0+lstm_8/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/ones_like
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dimФ
'lstm_8/lstm_cell_8/split/ReadVariableOpReadVariableOp0lstm_8_lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02)
'lstm_8/lstm_cell_8/split/ReadVariableOpѓ
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0/lstm_8/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_8/lstm_cell_8/splitЖ
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMulК
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_1К
lstm_8/lstm_cell_8/MatMul_2MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_2К
lstm_8/lstm_cell_8/MatMul_3MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_3
$lstm_8/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_8/lstm_cell_8/split_1/split_dimЦ
)lstm_8/lstm_cell_8/split_1/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_8/lstm_cell_8/split_1/ReadVariableOpы
lstm_8/lstm_cell_8/split_1Split-lstm_8/lstm_cell_8/split_1/split_dim:output:01lstm_8/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_8/lstm_cell_8/split_1П
lstm_8/lstm_cell_8/BiasAddBiasAdd#lstm_8/lstm_cell_8/MatMul:product:0#lstm_8/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAddХ
lstm_8/lstm_cell_8/BiasAdd_1BiasAdd%lstm_8/lstm_cell_8/MatMul_1:product:0#lstm_8/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_1Х
lstm_8/lstm_cell_8/BiasAdd_2BiasAdd%lstm_8/lstm_cell_8/MatMul_2:product:0#lstm_8/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_2Х
lstm_8/lstm_cell_8/BiasAdd_3BiasAdd%lstm_8/lstm_cell_8/MatMul_3:product:0#lstm_8/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_3Ї
lstm_8/lstm_cell_8/mulMullstm_8/zeros:output:0%lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mulЋ
lstm_8/lstm_cell_8/mul_1Mullstm_8/zeros:output:0%lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_1Ћ
lstm_8/lstm_cell_8/mul_2Mullstm_8/zeros:output:0%lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_2Ћ
lstm_8/lstm_cell_8/mul_3Mullstm_8/zeros:output:0%lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_3В
!lstm_8/lstm_cell_8/ReadVariableOpReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02#
!lstm_8/lstm_cell_8/ReadVariableOpЁ
&lstm_8/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_8/lstm_cell_8/strided_slice/stackЅ
(lstm_8/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_8/lstm_cell_8/strided_slice/stack_1Ѕ
(lstm_8/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_8/lstm_cell_8/strided_slice/stack_2ю
 lstm_8/lstm_cell_8/strided_sliceStridedSlice)lstm_8/lstm_cell_8/ReadVariableOp:value:0/lstm_8/lstm_cell_8/strided_slice/stack:output:01lstm_8/lstm_cell_8/strided_slice/stack_1:output:01lstm_8/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 lstm_8/lstm_cell_8/strided_sliceН
lstm_8/lstm_cell_8/MatMul_4MatMullstm_8/lstm_cell_8/mul:z:0)lstm_8/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_4З
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/BiasAdd:output:0%lstm_8/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add
lstm_8/lstm_cell_8/SigmoidSigmoidlstm_8/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/SigmoidЖ
#lstm_8/lstm_cell_8/ReadVariableOp_1ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_1Ѕ
(lstm_8/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_8/lstm_cell_8/strided_slice_1/stackЉ
*lstm_8/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_8/lstm_cell_8/strided_slice_1/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_1/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_1StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_1:value:01lstm_8/lstm_cell_8/strided_slice_1/stack:output:03lstm_8/lstm_cell_8/strided_slice_1/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_1С
lstm_8/lstm_cell_8/MatMul_5MatMullstm_8/lstm_cell_8/mul_1:z:0+lstm_8/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_5Н
lstm_8/lstm_cell_8/add_1AddV2%lstm_8/lstm_cell_8/BiasAdd_1:output:0%lstm_8/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_1
lstm_8/lstm_cell_8/Sigmoid_1Sigmoidlstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Sigmoid_1Ј
lstm_8/lstm_cell_8/mul_4Mul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_4Ж
#lstm_8/lstm_cell_8/ReadVariableOp_2ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_2Ѕ
(lstm_8/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_8/lstm_cell_8/strided_slice_2/stackЉ
*lstm_8/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_8/lstm_cell_8/strided_slice_2/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_2/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_2StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_2:value:01lstm_8/lstm_cell_8/strided_slice_2/stack:output:03lstm_8/lstm_cell_8/strided_slice_2/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_2С
lstm_8/lstm_cell_8/MatMul_6MatMullstm_8/lstm_cell_8/mul_2:z:0+lstm_8/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_6Н
lstm_8/lstm_cell_8/add_2AddV2%lstm_8/lstm_cell_8/BiasAdd_2:output:0%lstm_8/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_2
lstm_8/lstm_cell_8/ReluRelulstm_8/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/ReluД
lstm_8/lstm_cell_8/mul_5Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_5Ћ
lstm_8/lstm_cell_8/add_3AddV2lstm_8/lstm_cell_8/mul_4:z:0lstm_8/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_3Ж
#lstm_8/lstm_cell_8/ReadVariableOp_3ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_3Ѕ
(lstm_8/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(lstm_8/lstm_cell_8/strided_slice_3/stackЉ
*lstm_8/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_8/lstm_cell_8/strided_slice_3/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_3/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_3StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_3:value:01lstm_8/lstm_cell_8/strided_slice_3/stack:output:03lstm_8/lstm_cell_8/strided_slice_3/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_3С
lstm_8/lstm_cell_8/MatMul_7MatMullstm_8/lstm_cell_8/mul_3:z:0+lstm_8/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_7Н
lstm_8/lstm_cell_8/add_4AddV2%lstm_8/lstm_cell_8/BiasAdd_3:output:0%lstm_8/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_4
lstm_8/lstm_cell_8/Sigmoid_2Sigmoidlstm_8/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Sigmoid_2
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Relu_1И
lstm_8/lstm_cell_8/mul_6Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_6
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2&
$lstm_8/TensorArrayV2_1/element_shapeд
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterч
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_8_lstm_cell_8_split_readvariableop_resource2lstm_8_lstm_cell_8_split_1_readvariableop_resource*lstm_8_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_8_while_body_162128*$
condR
lstm_8_while_cond_162127*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_8/whileУ
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_8/strided_slice_3/stack
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2Ф
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_8/strided_slice_3
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/permС
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimeЈ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_12/MatMul/ReadVariableOpЇ
dense_12/MatMulMatMullstm_8/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/MatMulЇ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_12/BiasAdd/ReadVariableOpЅ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/ReluЈ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_13/MatMul/ReadVariableOpЃ
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_13/MatMulЇ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЅ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_13/BiasAddk
reshape_6/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_6/Shape
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stack
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2
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
reshape_6/Reshape/shape/2в
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shapeЄ
reshape_6/ReshapeReshapedense_13/BiasAdd:output:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_6/Reshapeь
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_8_lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muly
IdentityIdentityreshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp"^lstm_8/lstm_cell_8/ReadVariableOp$^lstm_8/lstm_cell_8/ReadVariableOp_1$^lstm_8/lstm_cell_8/ReadVariableOp_2$^lstm_8/lstm_cell_8/ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp(^lstm_8/lstm_cell_8/split/ReadVariableOp*^lstm_8/lstm_cell_8/split_1/ReadVariableOp^lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2F
!lstm_8/lstm_cell_8/ReadVariableOp!lstm_8/lstm_cell_8/ReadVariableOp2J
#lstm_8/lstm_cell_8/ReadVariableOp_1#lstm_8/lstm_cell_8/ReadVariableOp_12J
#lstm_8/lstm_cell_8/ReadVariableOp_2#lstm_8/lstm_cell_8/ReadVariableOp_22J
#lstm_8/lstm_cell_8/ReadVariableOp_3#lstm_8/lstm_cell_8/ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2R
'lstm_8/lstm_cell_8/split/ReadVariableOp'lstm_8/lstm_cell_8/split/ReadVariableOp2V
)lstm_8/lstm_cell_8/split_1/ReadVariableOp)lstm_8/lstm_cell_8/split_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

lstm_8_while_body_162128*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0:	I
:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0:	E
2lstm_8_while_lstm_cell_8_readvariableop_resource_0:	 
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorI
6lstm_8_while_lstm_cell_8_split_readvariableop_resource:	G
8lstm_8_while_lstm_cell_8_split_1_readvariableop_resource:	C
0lstm_8_while_lstm_cell_8_readvariableop_resource:	 Ђ'lstm_8/while/lstm_cell_8/ReadVariableOpЂ)lstm_8/while/lstm_cell_8/ReadVariableOp_1Ђ)lstm_8/while/lstm_cell_8/ReadVariableOp_2Ђ)lstm_8/while/lstm_cell_8/ReadVariableOp_3Ђ-lstm_8/while/lstm_cell_8/split/ReadVariableOpЂ/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpб
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem
(lstm_8/while/lstm_cell_8/ones_like/ShapeShapelstm_8_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm_8/while/lstm_cell_8/ones_like/Shape
(lstm_8/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm_8/while/lstm_cell_8/ones_like/Constш
"lstm_8/while/lstm_cell_8/ones_likeFill1lstm_8/while/lstm_cell_8/ones_like/Shape:output:01lstm_8/while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/ones_like
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dimи
-lstm_8/while/lstm_cell_8/split/ReadVariableOpReadVariableOp8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02/
-lstm_8/while/lstm_cell_8/split/ReadVariableOp
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:05lstm_8/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2 
lstm_8/while/lstm_cell_8/splitр
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_8/while/lstm_cell_8/MatMulф
!lstm_8/while/lstm_cell_8/MatMul_1MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_1ф
!lstm_8/while/lstm_cell_8/MatMul_2MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_2ф
!lstm_8/while/lstm_cell_8/MatMul_3MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_3
*lstm_8/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_8/while/lstm_cell_8/split_1/split_dimк
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp
 lstm_8/while/lstm_cell_8/split_1Split3lstm_8/while/lstm_cell_8/split_1/split_dim:output:07lstm_8/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2"
 lstm_8/while/lstm_cell_8/split_1з
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd)lstm_8/while/lstm_cell_8/MatMul:product:0)lstm_8/while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/while/lstm_cell_8/BiasAddн
"lstm_8/while/lstm_cell_8/BiasAdd_1BiasAdd+lstm_8/while/lstm_cell_8/MatMul_1:product:0)lstm_8/while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_1н
"lstm_8/while/lstm_cell_8/BiasAdd_2BiasAdd+lstm_8/while/lstm_cell_8/MatMul_2:product:0)lstm_8/while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_2н
"lstm_8/while/lstm_cell_8/BiasAdd_3BiasAdd+lstm_8/while/lstm_cell_8/MatMul_3:product:0)lstm_8/while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_3О
lstm_8/while/lstm_cell_8/mulMullstm_8_while_placeholder_2+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/mulТ
lstm_8/while/lstm_cell_8/mul_1Mullstm_8_while_placeholder_2+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_1Т
lstm_8/while/lstm_cell_8/mul_2Mullstm_8_while_placeholder_2+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_2Т
lstm_8/while/lstm_cell_8/mul_3Mullstm_8_while_placeholder_2+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_3Ц
'lstm_8/while/lstm_cell_8/ReadVariableOpReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02)
'lstm_8/while/lstm_cell_8/ReadVariableOp­
,lstm_8/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_8/while/lstm_cell_8/strided_slice/stackБ
.lstm_8/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_8/while/lstm_cell_8/strided_slice/stack_1Б
.lstm_8/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_8/while/lstm_cell_8/strided_slice/stack_2
&lstm_8/while/lstm_cell_8/strided_sliceStridedSlice/lstm_8/while/lstm_cell_8/ReadVariableOp:value:05lstm_8/while/lstm_cell_8/strided_slice/stack:output:07lstm_8/while/lstm_cell_8/strided_slice/stack_1:output:07lstm_8/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_8/while/lstm_cell_8/strided_sliceе
!lstm_8/while/lstm_cell_8/MatMul_4MatMul lstm_8/while/lstm_cell_8/mul:z:0/lstm_8/while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_4Я
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/BiasAdd:output:0+lstm_8/while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/addЃ
 lstm_8/while/lstm_cell_8/SigmoidSigmoid lstm_8/while/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/while/lstm_cell_8/SigmoidЪ
)lstm_8/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_1Б
.lstm_8/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_8/while/lstm_cell_8/strided_slice_1/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_1StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_1:value:07lstm_8/while/lstm_cell_8/strided_slice_1/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_1/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_1й
!lstm_8/while/lstm_cell_8/MatMul_5MatMul"lstm_8/while/lstm_cell_8/mul_1:z:01lstm_8/while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_5е
lstm_8/while/lstm_cell_8/add_1AddV2+lstm_8/while/lstm_cell_8/BiasAdd_1:output:0+lstm_8/while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_1Љ
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/Sigmoid_1Н
lstm_8/while/lstm_cell_8/mul_4Mul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_4Ъ
)lstm_8/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_2Б
.lstm_8/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_8/while/lstm_cell_8/strided_slice_2/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_2StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_2:value:07lstm_8/while/lstm_cell_8/strided_slice_2/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_2/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_2й
!lstm_8/while/lstm_cell_8/MatMul_6MatMul"lstm_8/while/lstm_cell_8/mul_2:z:01lstm_8/while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_6е
lstm_8/while/lstm_cell_8/add_2AddV2+lstm_8/while/lstm_cell_8/BiasAdd_2:output:0+lstm_8/while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_2
lstm_8/while/lstm_cell_8/ReluRelu"lstm_8/while/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/ReluЬ
lstm_8/while/lstm_cell_8/mul_5Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_5У
lstm_8/while/lstm_cell_8/add_3AddV2"lstm_8/while/lstm_cell_8/mul_4:z:0"lstm_8/while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_3Ъ
)lstm_8/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_3Б
.lstm_8/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_8/while/lstm_cell_8/strided_slice_3/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_3StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_3:value:07lstm_8/while/lstm_cell_8/strided_slice_3/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_3/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_3й
!lstm_8/while/lstm_cell_8/MatMul_7MatMul"lstm_8/while/lstm_cell_8/mul_3:z:01lstm_8/while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_7е
lstm_8/while/lstm_cell_8/add_4AddV2+lstm_8/while/lstm_cell_8/BiasAdd_3:output:0+lstm_8/while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_4Љ
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid"lstm_8/while/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/Sigmoid_2 
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_8/while/lstm_cell_8/Relu_1а
lstm_8/while/lstm_cell_8/mul_6Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_6
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_6:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/IdentityЁ
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2Ж
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3Ј
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_6:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/Identity_4Ј
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_3:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/Identity_5ј
lstm_8/while/NoOpNoOp(^lstm_8/while/lstm_cell_8/ReadVariableOp*^lstm_8/while/lstm_cell_8/ReadVariableOp_1*^lstm_8/while/lstm_cell_8/ReadVariableOp_2*^lstm_8/while/lstm_cell_8/ReadVariableOp_3.^lstm_8/while/lstm_cell_8/split/ReadVariableOp0^lstm_8/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_8/while/NoOp"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"f
0lstm_8_while_lstm_cell_8_readvariableop_resource2lstm_8_while_lstm_cell_8_readvariableop_resource_0"v
8lstm_8_while_lstm_cell_8_split_1_readvariableop_resource:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0"r
6lstm_8_while_lstm_cell_8_split_readvariableop_resource8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0"Ф
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2R
'lstm_8/while/lstm_cell_8/ReadVariableOp'lstm_8/while/lstm_cell_8/ReadVariableOp2V
)lstm_8/while/lstm_cell_8/ReadVariableOp_1)lstm_8/while/lstm_cell_8/ReadVariableOp_12V
)lstm_8/while/lstm_cell_8/ReadVariableOp_2)lstm_8/while/lstm_cell_8/ReadVariableOp_22V
)lstm_8/while/lstm_cell_8/ReadVariableOp_3)lstm_8/while/lstm_cell_8/ReadVariableOp_32^
-lstm_8/while/lstm_cell_8/split/ReadVariableOp-lstm_8/while/lstm_cell_8/split/ReadVariableOp2b
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp: 
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

a
E__inference_reshape_6_layer_call_and_return_conditional_losses_161463

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
ё

)__inference_dense_13_layer_call_fn_163839

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallє
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
GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1614442
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
П%
м
while_body_160811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_160835_0:	)
while_lstm_cell_8_160837_0:	-
while_lstm_cell_8_160839_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_160835:	'
while_lstm_cell_8_160837:	+
while_lstm_cell_8_160839:	 Ђ)while/lstm_cell_8/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_160835_0while_lstm_cell_8_160837_0while_lstm_cell_8_160839_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1607332+
)while/lstm_cell_8/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_160835while_lstm_cell_8_160835_0"6
while_lstm_cell_8_160837while_lstm_cell_8_160837_0"6
while_lstm_cell_8_160839while_lstm_cell_8_160839_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
Џ
г
%sequential_4_lstm_8_while_cond_160226D
@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counterJ
Fsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations)
%sequential_4_lstm_8_while_placeholder+
'sequential_4_lstm_8_while_placeholder_1+
'sequential_4_lstm_8_while_placeholder_2+
'sequential_4_lstm_8_while_placeholder_3F
Bsequential_4_lstm_8_while_less_sequential_4_lstm_8_strided_slice_1\
Xsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_160226___redundant_placeholder0\
Xsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_160226___redundant_placeholder1\
Xsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_160226___redundant_placeholder2\
Xsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_160226___redundant_placeholder3&
"sequential_4_lstm_8_while_identity
д
sequential_4/lstm_8/while/LessLess%sequential_4_lstm_8_while_placeholderBsequential_4_lstm_8_while_less_sequential_4_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_4/lstm_8/while/Less
"sequential_4/lstm_8/while/IdentityIdentity"sequential_4/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_4/lstm_8/while/Identity"Q
"sequential_4_lstm_8_while_identity+sequential_4/lstm_8/while/Identity:output:0*(
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
е
У
while_cond_163040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163040___redundant_placeholder04
0while_while_cond_163040___redundant_placeholder14
0while_while_cond_163040___redundant_placeholder24
0while_while_cond_163040___redundant_placeholder3
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
М
Ж
'__inference_lstm_8_layer_call_fn_163778
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1608862
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
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
аQ
О
B__inference_lstm_8_layer_call_and_return_conditional_losses_160886

inputs%
lstm_cell_8_160798:	!
lstm_cell_8_160800:	%
lstm_cell_8_160802:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂ#lstm_cell_8/StatefulPartitionedCallЂwhileD
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
 :џџџџџџџџџџџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_160798lstm_cell_8_160800lstm_cell_8_160802*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1607332%
#lstm_cell_8/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_160798lstm_cell_8_160800lstm_cell_8_160802*
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
bodyR
while_body_160811*
condR
while_cond_160810*K
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
runtimeЮ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_8_160798*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityК
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
F
*__inference_reshape_6_layer_call_fn_163857

inputs
identityЧ
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
GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_1614632
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
І

ѕ
D__inference_dense_13_layer_call_and_return_conditional_losses_163830

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

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
П%
м
while_body_160514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_160538_0:	)
while_lstm_cell_8_160540_0:	-
while_lstm_cell_8_160542_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_160538:	'
while_lstm_cell_8_160540:	+
while_lstm_cell_8_160542:	 Ђ)while/lstm_cell_8/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_160538_0while_lstm_cell_8_160540_0while_lstm_cell_8_160542_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1605002+
)while/lstm_cell_8/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_160538while_lstm_cell_8_160538_0"6
while_lstm_cell_8_160540while_lstm_cell_8_160540_0"6
while_lstm_cell_8_160542while_lstm_cell_8_160542_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
ЃR
ц
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_160500

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6й
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
э!
ѓ
H__inference_sequential_4_layer_call_and_return_conditional_losses_161957
input_5 
lstm_8_161932:	
lstm_8_161934:	 
lstm_8_161936:	 !
dense_12_161939:  
dense_12_161941: !
dense_13_161944: 
dense_13_161946:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂlstm_8/StatefulPartitionedCallЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_8_161932lstm_8_161934lstm_8_161936*
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1614092 
lstm_8/StatefulPartitionedCallЕ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_12_161939dense_12_161941*
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
GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1614282"
 dense_12/StatefulPartitionedCallЗ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_161944dense_13_161946*
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
GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1614442"
 dense_13/StatefulPartitionedCallў
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_1614632
reshape_6/PartitionedCallЩ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_8_161932*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/mul
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityѓ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
у	
Ї
-__inference_sequential_4_layer_call_fn_161489
input_5
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1614722
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
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
р	
І
-__inference_sequential_4_layer_call_fn_162631

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallН
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1614722
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
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у	
Ї
-__inference_sequential_4_layer_call_fn_161929
input_5
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1618932
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
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
Ф~
	
while_body_161276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ђ
while/lstm_cell_8/mulMulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulІ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1І
while/lstm_cell_8/mul_2Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2І
while/lstm_cell_8/mul_3Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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

a
E__inference_reshape_6_layer_call_and_return_conditional_losses_163852

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
§{
ю
"__inference__traced_restore_164303
file_prefix2
 assignvariableop_dense_12_kernel:  .
 assignvariableop_1_dense_12_bias: 4
"assignvariableop_2_dense_13_kernel: .
 assignvariableop_3_dense_13_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
,assignvariableop_9_lstm_8_lstm_cell_8_kernel:	J
7assignvariableop_10_lstm_8_lstm_cell_8_recurrent_kernel:	 :
+assignvariableop_11_lstm_8_lstm_cell_8_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_12_kernel_m:  6
(assignvariableop_15_adam_dense_12_bias_m: <
*assignvariableop_16_adam_dense_13_kernel_m: 6
(assignvariableop_17_adam_dense_13_bias_m:G
4assignvariableop_18_adam_lstm_8_lstm_cell_8_kernel_m:	Q
>assignvariableop_19_adam_lstm_8_lstm_cell_8_recurrent_kernel_m:	 A
2assignvariableop_20_adam_lstm_8_lstm_cell_8_bias_m:	<
*assignvariableop_21_adam_dense_12_kernel_v:  6
(assignvariableop_22_adam_dense_12_bias_v: <
*assignvariableop_23_adam_dense_13_kernel_v: 6
(assignvariableop_24_adam_dense_13_bias_v:G
4assignvariableop_25_adam_lstm_8_lstm_cell_8_kernel_v:	Q
>assignvariableop_26_adam_lstm_8_lstm_cell_8_recurrent_kernel_v:	 A
2assignvariableop_27_adam_lstm_8_lstm_cell_8_bias_v:	
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
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9Б
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_8_lstm_cell_8_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10П
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_8_lstm_cell_8_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Г
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_8_lstm_cell_8_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_12_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15А
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_12_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_13_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_13_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18М
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_8_lstm_cell_8_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ц
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_lstm_8_lstm_cell_8_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20К
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_lstm_8_lstm_cell_8_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_12_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_12_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_13_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_13_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25М
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_8_lstm_cell_8_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ц
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_lstm_8_lstm_cell_8_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27К
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_lstm_8_lstm_cell_8_bias_vIdentity_27:output:0"/device:CPU:0*
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
ЈB
а
__inference__traced_save_164209
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesЪ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Щ: :  : : :: : : : : :	:	 :: : :  : : ::	:	 ::  : : ::	:	 :: 2(
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
:	:%!

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
:	:%!

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
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: 
е
У
while_cond_163315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163315___redundant_placeholder04
0while_while_cond_163315___redundant_placeholder14
0while_while_cond_163315___redundant_placeholder24
0while_while_cond_163315___redundant_placeholder3
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
Д
ѕ
,__inference_lstm_cell_8_layer_call_fn_164091

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1607332
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
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
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

ѕ
D__inference_dense_12_layer_call_and_return_conditional_losses_161428

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
е
У
while_cond_161275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_161275___redundant_placeholder04
0while_while_cond_161275___redundant_placeholder14
0while_while_cond_161275___redundant_placeholder24
0while_while_cond_161275___redundant_placeholder3
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
Ј

Я
lstm_8_while_cond_162424*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_162424___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_162424___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_162424___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_162424___redundant_placeholder3
lstm_8_while_identity

lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
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
ІЭ

B__inference_lstm_8_layer_call_and_return_conditional_losses_163206
inputs_0<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileF
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
 :џџџџџџџџџџџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like{
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout/ConstЏ
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout/Shapeѕ
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Щ?22
0lstm_cell_8/dropout/random_uniform/RandomUniform
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2$
"lstm_cell_8/dropout/GreaterEqual/yю
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_cell_8/dropout/GreaterEqualЃ
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/CastЊ
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul_1
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_1/ConstЕ
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_1/Shapeќ
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed224
2lstm_cell_8/dropout_1/random_uniform/RandomUniform
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_1/GreaterEqual/yі
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_1/GreaterEqualЉ
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/CastВ
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul_1
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_2/ConstЕ
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_2/Shape§
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2њ 24
2lstm_cell_8/dropout_2/random_uniform/RandomUniform
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_2/GreaterEqual/yі
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_2/GreaterEqualЉ
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/CastВ
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul_1
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_3/ConstЕ
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_3/Shape§
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЌЬ 24
2lstm_cell_8/dropout_3/random_uniform/RandomUniform
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_3/GreaterEqual/yі
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_3/GreaterEqualЉ
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/CastВ
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul_1|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_163041*
condR
while_cond_163040*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
МА
	
while_body_163591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2!
while/lstm_cell_8/dropout/ConstЧ
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/dropout/Mul
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_8/dropout/Shape
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2СУ28
6while/lstm_cell_8/dropout/random_uniform/RandomUniform
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2*
(while/lstm_cell_8/dropout/GreaterEqual/y
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&while/lstm_cell_8/dropout/GreaterEqualЕ
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_8/dropout/CastТ
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout/Mul_1
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_1/ConstЭ
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_1/Mul
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_1/Shape
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2мЈё2:
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_1/GreaterEqual/y
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_1/GreaterEqualЛ
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_1/CastЪ
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_1/Mul_1
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_2/ConstЭ
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_2/Mul
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_2/Shape
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ё2:
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_2/GreaterEqual/y
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_2/GreaterEqualЛ
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_2/CastЪ
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_2/Mul_1
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_3/ConstЭ
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_3/Mul
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_3/Shape
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ўфБ2:
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_3/GreaterEqual/y
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_3/GreaterEqualЛ
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_3/CastЪ
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_3/Mul_1
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ё
while/lstm_cell_8/mulMulwhile_placeholder_2#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulЇ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1Ї
while/lstm_cell_8/mul_2Mulwhile_placeholder_2%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2Ї
while/lstm_cell_8/mul_3Mulwhile_placeholder_2%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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
Д
ѕ
,__inference_lstm_cell_8_layer_call_fn_164074

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1605002
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
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
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
ёЬ

B__inference_lstm_8_layer_call_and_return_conditional_losses_161835

inputs<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileD
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
:џџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like{
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout/ConstЏ
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout/Shapeі
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Жb22
0lstm_cell_8/dropout/random_uniform/RandomUniform
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2$
"lstm_cell_8/dropout/GreaterEqual/yю
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_cell_8/dropout/GreaterEqualЃ
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/CastЊ
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul_1
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_1/ConstЕ
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_1/Shapeќ
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2хv24
2lstm_cell_8/dropout_1/random_uniform/RandomUniform
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_1/GreaterEqual/yі
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_1/GreaterEqualЉ
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/CastВ
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul_1
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_2/ConstЕ
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_2/Shape§
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ѕр24
2lstm_cell_8/dropout_2/random_uniform/RandomUniform
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_2/GreaterEqual/yі
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_2/GreaterEqualЉ
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/CastВ
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul_1
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_3/ConstЕ
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_3/Shape§
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2сЧђ24
2lstm_cell_8/dropout_3/random_uniform/RandomUniform
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_3/GreaterEqual/yі
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_3/GreaterEqualЉ
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/CastВ
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul_1|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_161670*
condR
while_cond_161669*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з

B__inference_lstm_8_layer_call_and_return_conditional_losses_161409

inputs<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileD
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
:џџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_161276*
condR
while_cond_161275*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р	
І
-__inference_sequential_4_layer_call_fn_162650

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallН
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1618932
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
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ГR
ш
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_163944

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6й
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
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
v
ц
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_160733

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
 *UUе?2
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
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2НА2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ј2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2дЁЉ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_3/Shapeй
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЗЦк2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6й
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
Є
Д
'__inference_lstm_8_layer_call_fn_163789

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCallџ
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1614092
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
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ѕ
D__inference_dense_12_layer_call_and_return_conditional_losses_163811

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
Г	

$__inference_signature_wrapper_162018
input_5
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 **
f%R#
!__inference__wrapped_model_1603762
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
%:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
е
У
while_cond_160513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_160513___redundant_placeholder04
0while_while_cond_160513___redundant_placeholder14
0while_while_cond_160513___redundant_placeholder24
0while_while_cond_160513___redundant_placeholder3
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
ё

)__inference_dense_12_layer_call_fn_163820

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallє
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
GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1614282
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
лѕ
й
!__inference__wrapped_model_160376
input_5P
=sequential_4_lstm_8_lstm_cell_8_split_readvariableop_resource:	N
?sequential_4_lstm_8_lstm_cell_8_split_1_readvariableop_resource:	J
7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource:	 F
4sequential_4_dense_12_matmul_readvariableop_resource:  C
5sequential_4_dense_12_biasadd_readvariableop_resource: F
4sequential_4_dense_13_matmul_readvariableop_resource: C
5sequential_4_dense_13_biasadd_readvariableop_resource:
identityЂ,sequential_4/dense_12/BiasAdd/ReadVariableOpЂ+sequential_4/dense_12/MatMul/ReadVariableOpЂ,sequential_4/dense_13/BiasAdd/ReadVariableOpЂ+sequential_4/dense_13/MatMul/ReadVariableOpЂ.sequential_4/lstm_8/lstm_cell_8/ReadVariableOpЂ0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_1Ђ0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_2Ђ0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_3Ђ4sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOpЂ6sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOpЂsequential_4/lstm_8/whilem
sequential_4/lstm_8/ShapeShapeinput_5*
T0*
_output_shapes
:2
sequential_4/lstm_8/Shape
'sequential_4/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_4/lstm_8/strided_slice/stack 
)sequential_4/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_8/strided_slice/stack_1 
)sequential_4/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_8/strided_slice/stack_2к
!sequential_4/lstm_8/strided_sliceStridedSlice"sequential_4/lstm_8/Shape:output:00sequential_4/lstm_8/strided_slice/stack:output:02sequential_4/lstm_8/strided_slice/stack_1:output:02sequential_4/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_4/lstm_8/strided_slice
sequential_4/lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_4/lstm_8/zeros/mul/yМ
sequential_4/lstm_8/zeros/mulMul*sequential_4/lstm_8/strided_slice:output:0(sequential_4/lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_8/zeros/mul
 sequential_4/lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_4/lstm_8/zeros/Less/yЗ
sequential_4/lstm_8/zeros/LessLess!sequential_4/lstm_8/zeros/mul:z:0)sequential_4/lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_8/zeros/Less
"sequential_4/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_4/lstm_8/zeros/packed/1г
 sequential_4/lstm_8/zeros/packedPack*sequential_4/lstm_8/strided_slice:output:0+sequential_4/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_4/lstm_8/zeros/packed
sequential_4/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_4/lstm_8/zeros/ConstХ
sequential_4/lstm_8/zerosFill)sequential_4/lstm_8/zeros/packed:output:0(sequential_4/lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_4/lstm_8/zeros
!sequential_4/lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_4/lstm_8/zeros_1/mul/yТ
sequential_4/lstm_8/zeros_1/mulMul*sequential_4/lstm_8/strided_slice:output:0*sequential_4/lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_8/zeros_1/mul
"sequential_4/lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_4/lstm_8/zeros_1/Less/yП
 sequential_4/lstm_8/zeros_1/LessLess#sequential_4/lstm_8/zeros_1/mul:z:0+sequential_4/lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_8/zeros_1/Less
$sequential_4/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_4/lstm_8/zeros_1/packed/1й
"sequential_4/lstm_8/zeros_1/packedPack*sequential_4/lstm_8/strided_slice:output:0-sequential_4/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_4/lstm_8/zeros_1/packed
!sequential_4/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_4/lstm_8/zeros_1/ConstЭ
sequential_4/lstm_8/zeros_1Fill+sequential_4/lstm_8/zeros_1/packed:output:0*sequential_4/lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_4/lstm_8/zeros_1
"sequential_4/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_4/lstm_8/transpose/permЗ
sequential_4/lstm_8/transpose	Transposeinput_5+sequential_4/lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_4/lstm_8/transpose
sequential_4/lstm_8/Shape_1Shape!sequential_4/lstm_8/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_8/Shape_1 
)sequential_4/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_8/strided_slice_1/stackЄ
+sequential_4/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_1/stack_1Є
+sequential_4/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_1/stack_2ц
#sequential_4/lstm_8/strided_slice_1StridedSlice$sequential_4/lstm_8/Shape_1:output:02sequential_4/lstm_8/strided_slice_1/stack:output:04sequential_4/lstm_8/strided_slice_1/stack_1:output:04sequential_4/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_1­
/sequential_4/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ21
/sequential_4/lstm_8/TensorArrayV2/element_shape
!sequential_4/lstm_8/TensorArrayV2TensorListReserve8sequential_4/lstm_8/TensorArrayV2/element_shape:output:0,sequential_4/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_4/lstm_8/TensorArrayV2ч
Isequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Isequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
;sequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_8/transpose:y:0Rsequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor 
)sequential_4/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_8/strided_slice_2/stackЄ
+sequential_4/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_2/stack_1Є
+sequential_4/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_2/stack_2є
#sequential_4/lstm_8/strided_slice_2StridedSlice!sequential_4/lstm_8/transpose:y:02sequential_4/lstm_8/strided_slice_2/stack:output:04sequential_4/lstm_8/strided_slice_2/stack_1:output:04sequential_4/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_2Д
/sequential_4/lstm_8/lstm_cell_8/ones_like/ShapeShape"sequential_4/lstm_8/zeros:output:0*
T0*
_output_shapes
:21
/sequential_4/lstm_8/lstm_cell_8/ones_like/ShapeЇ
/sequential_4/lstm_8/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/sequential_4/lstm_8/lstm_cell_8/ones_like/Const
)sequential_4/lstm_8/lstm_cell_8/ones_likeFill8sequential_4/lstm_8/lstm_cell_8/ones_like/Shape:output:08sequential_4/lstm_8/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/ones_likeЄ
/sequential_4/lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_4/lstm_8/lstm_cell_8/split/split_dimы
4sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOpReadVariableOp=sequential_4_lstm_8_lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype026
4sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOpЇ
%sequential_4/lstm_8/lstm_cell_8/splitSplit8sequential_4/lstm_8/lstm_cell_8/split/split_dim:output:0<sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2'
%sequential_4/lstm_8/lstm_cell_8/splitъ
&sequential_4/lstm_8/lstm_cell_8/MatMulMatMul,sequential_4/lstm_8/strided_slice_2:output:0.sequential_4/lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_4/lstm_8/lstm_cell_8/MatMulю
(sequential_4/lstm_8/lstm_cell_8/MatMul_1MatMul,sequential_4/lstm_8/strided_slice_2:output:0.sequential_4/lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_1ю
(sequential_4/lstm_8/lstm_cell_8/MatMul_2MatMul,sequential_4/lstm_8/strided_slice_2:output:0.sequential_4/lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_2ю
(sequential_4/lstm_8/lstm_cell_8/MatMul_3MatMul,sequential_4/lstm_8/strided_slice_2:output:0.sequential_4/lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_3Ј
1sequential_4/lstm_8/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_4/lstm_8/lstm_cell_8/split_1/split_dimэ
6sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOpReadVariableOp?sequential_4_lstm_8_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype028
6sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOp
'sequential_4/lstm_8/lstm_cell_8/split_1Split:sequential_4/lstm_8/lstm_cell_8/split_1/split_dim:output:0>sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2)
'sequential_4/lstm_8/lstm_cell_8/split_1ѓ
'sequential_4/lstm_8/lstm_cell_8/BiasAddBiasAdd0sequential_4/lstm_8/lstm_cell_8/MatMul:product:00sequential_4/lstm_8/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_4/lstm_8/lstm_cell_8/BiasAddљ
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_1BiasAdd2sequential_4/lstm_8/lstm_cell_8/MatMul_1:product:00sequential_4/lstm_8/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_1љ
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_2BiasAdd2sequential_4/lstm_8/lstm_cell_8/MatMul_2:product:00sequential_4/lstm_8/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_2љ
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_3BiasAdd2sequential_4/lstm_8/lstm_cell_8/MatMul_3:product:00sequential_4/lstm_8/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/BiasAdd_3л
#sequential_4/lstm_8/lstm_cell_8/mulMul"sequential_4/lstm_8/zeros:output:02sequential_4/lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#sequential_4/lstm_8/lstm_cell_8/mulп
%sequential_4/lstm_8/lstm_cell_8/mul_1Mul"sequential_4/lstm_8/zeros:output:02sequential_4/lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_1п
%sequential_4/lstm_8/lstm_cell_8/mul_2Mul"sequential_4/lstm_8/zeros:output:02sequential_4/lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_2п
%sequential_4/lstm_8/lstm_cell_8/mul_3Mul"sequential_4/lstm_8/zeros:output:02sequential_4/lstm_8/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_3й
.sequential_4/lstm_8/lstm_cell_8/ReadVariableOpReadVariableOp7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype020
.sequential_4/lstm_8/lstm_cell_8/ReadVariableOpЛ
3sequential_4/lstm_8/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential_4/lstm_8/lstm_cell_8/strided_slice/stackП
5sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_1П
5sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_2М
-sequential_4/lstm_8/lstm_cell_8/strided_sliceStridedSlice6sequential_4/lstm_8/lstm_cell_8/ReadVariableOp:value:0<sequential_4/lstm_8/lstm_cell_8/strided_slice/stack:output:0>sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_1:output:0>sequential_4/lstm_8/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2/
-sequential_4/lstm_8/lstm_cell_8/strided_sliceё
(sequential_4/lstm_8/lstm_cell_8/MatMul_4MatMul'sequential_4/lstm_8/lstm_cell_8/mul:z:06sequential_4/lstm_8/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_4ы
#sequential_4/lstm_8/lstm_cell_8/addAddV20sequential_4/lstm_8/lstm_cell_8/BiasAdd:output:02sequential_4/lstm_8/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#sequential_4/lstm_8/lstm_cell_8/addИ
'sequential_4/lstm_8/lstm_cell_8/SigmoidSigmoid'sequential_4/lstm_8/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_4/lstm_8/lstm_cell_8/Sigmoidн
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_1ReadVariableOp7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype022
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_1П
5sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stackУ
7sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_1У
7sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_2Ш
/sequential_4/lstm_8/lstm_cell_8/strided_slice_1StridedSlice8sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_1:value:0>sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_1:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_4/lstm_8/lstm_cell_8/strided_slice_1ѕ
(sequential_4/lstm_8/lstm_cell_8/MatMul_5MatMul)sequential_4/lstm_8/lstm_cell_8/mul_1:z:08sequential_4/lstm_8/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_5ё
%sequential_4/lstm_8/lstm_cell_8/add_1AddV22sequential_4/lstm_8/lstm_cell_8/BiasAdd_1:output:02sequential_4/lstm_8/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/add_1О
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_1Sigmoid)sequential_4/lstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_1м
%sequential_4/lstm_8/lstm_cell_8/mul_4Mul-sequential_4/lstm_8/lstm_cell_8/Sigmoid_1:y:0$sequential_4/lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_4н
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_2ReadVariableOp7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype022
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_2П
5sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stackУ
7sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_1У
7sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_2Ш
/sequential_4/lstm_8/lstm_cell_8/strided_slice_2StridedSlice8sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_2:value:0>sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_1:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_4/lstm_8/lstm_cell_8/strided_slice_2ѕ
(sequential_4/lstm_8/lstm_cell_8/MatMul_6MatMul)sequential_4/lstm_8/lstm_cell_8/mul_2:z:08sequential_4/lstm_8/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_6ё
%sequential_4/lstm_8/lstm_cell_8/add_2AddV22sequential_4/lstm_8/lstm_cell_8/BiasAdd_2:output:02sequential_4/lstm_8/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/add_2Б
$sequential_4/lstm_8/lstm_cell_8/ReluRelu)sequential_4/lstm_8/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_4/lstm_8/lstm_cell_8/Reluш
%sequential_4/lstm_8/lstm_cell_8/mul_5Mul+sequential_4/lstm_8/lstm_cell_8/Sigmoid:y:02sequential_4/lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_5п
%sequential_4/lstm_8/lstm_cell_8/add_3AddV2)sequential_4/lstm_8/lstm_cell_8/mul_4:z:0)sequential_4/lstm_8/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/add_3н
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_3ReadVariableOp7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype022
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_3П
5sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   27
5sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stackУ
7sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_1У
7sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_2Ш
/sequential_4/lstm_8/lstm_cell_8/strided_slice_3StridedSlice8sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_3:value:0>sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_1:output:0@sequential_4/lstm_8/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_4/lstm_8/lstm_cell_8/strided_slice_3ѕ
(sequential_4/lstm_8/lstm_cell_8/MatMul_7MatMul)sequential_4/lstm_8/lstm_cell_8/mul_3:z:08sequential_4/lstm_8/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_7ё
%sequential_4/lstm_8/lstm_cell_8/add_4AddV22sequential_4/lstm_8/lstm_cell_8/BiasAdd_3:output:02sequential_4/lstm_8/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/add_4О
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_2Sigmoid)sequential_4/lstm_8/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_2Е
&sequential_4/lstm_8/lstm_cell_8/Relu_1Relu)sequential_4/lstm_8/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_4/lstm_8/lstm_cell_8/Relu_1ь
%sequential_4/lstm_8/lstm_cell_8/mul_6Mul-sequential_4/lstm_8/lstm_cell_8/Sigmoid_2:y:04sequential_4/lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_4/lstm_8/lstm_cell_8/mul_6З
1sequential_4/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    23
1sequential_4/lstm_8/TensorArrayV2_1/element_shape
#sequential_4/lstm_8/TensorArrayV2_1TensorListReserve:sequential_4/lstm_8/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_4/lstm_8/TensorArrayV2_1v
sequential_4/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_8/timeЇ
,sequential_4/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,sequential_4/lstm_8/while/maximum_iterations
&sequential_4/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_4/lstm_8/while/loop_counterЊ
sequential_4/lstm_8/whileWhile/sequential_4/lstm_8/while/loop_counter:output:05sequential_4/lstm_8/while/maximum_iterations:output:0!sequential_4/lstm_8/time:output:0,sequential_4/lstm_8/TensorArrayV2_1:handle:0"sequential_4/lstm_8/zeros:output:0$sequential_4/lstm_8/zeros_1:output:0,sequential_4/lstm_8/strided_slice_1:output:0Ksequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_4_lstm_8_lstm_cell_8_split_readvariableop_resource?sequential_4_lstm_8_lstm_cell_8_split_1_readvariableop_resource7sequential_4_lstm_8_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_4_lstm_8_while_body_160227*1
cond)R'
%sequential_4_lstm_8_while_cond_160226*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_4/lstm_8/whileн
Dsequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2F
Dsequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeИ
6sequential_4/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_8/while:output:3Msequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype028
6sequential_4/lstm_8/TensorArrayV2Stack/TensorListStackЉ
)sequential_4/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2+
)sequential_4/lstm_8/strided_slice_3/stackЄ
+sequential_4/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_4/lstm_8/strided_slice_3/stack_1Є
+sequential_4/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_3/stack_2
#sequential_4/lstm_8/strided_slice_3StridedSlice?sequential_4/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_8/strided_slice_3/stack:output:04sequential_4/lstm_8/strided_slice_3/stack_1:output:04sequential_4/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_3Ё
$sequential_4/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_4/lstm_8/transpose_1/permѕ
sequential_4/lstm_8/transpose_1	Transpose?sequential_4/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2!
sequential_4/lstm_8/transpose_1
sequential_4/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_8/runtimeЯ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpл
sequential_4/dense_12/MatMulMatMul,sequential_4/lstm_8/strided_slice_3:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_4/dense_12/MatMulЮ
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpй
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_4/dense_12/BiasAdd
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_4/dense_12/ReluЯ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOpз
sequential_4/dense_13/MatMulMatMul(sequential_4/dense_12/Relu:activations:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_4/dense_13/MatMulЮ
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpй
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_4/dense_13/BiasAdd
sequential_4/reshape_6/ShapeShape&sequential_4/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_4/reshape_6/ShapeЂ
*sequential_4/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_6/strided_slice/stackІ
,sequential_4/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_6/strided_slice/stack_1І
,sequential_4/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_6/strided_slice/stack_2ь
$sequential_4/reshape_6/strided_sliceStridedSlice%sequential_4/reshape_6/Shape:output:03sequential_4/reshape_6/strided_slice/stack:output:05sequential_4/reshape_6/strided_slice/stack_1:output:05sequential_4/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_6/strided_slice
&sequential_4/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_6/Reshape/shape/1
&sequential_4/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_6/Reshape/shape/2
$sequential_4/reshape_6/Reshape/shapePack-sequential_4/reshape_6/strided_slice:output:0/sequential_4/reshape_6/Reshape/shape/1:output:0/sequential_4/reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_6/Reshape/shapeи
sequential_4/reshape_6/ReshapeReshape&sequential_4/dense_13/BiasAdd:output:0-sequential_4/reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_4/reshape_6/Reshape
IdentityIdentity'sequential_4/reshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityо
NoOpNoOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp/^sequential_4/lstm_8/lstm_cell_8/ReadVariableOp1^sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_11^sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_21^sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_35^sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOp7^sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOp^sequential_4/lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp2`
.sequential_4/lstm_8/lstm_cell_8/ReadVariableOp.sequential_4/lstm_8/lstm_cell_8/ReadVariableOp2d
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_10sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_12d
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_20sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_22d
0sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_30sequential_4/lstm_8/lstm_cell_8/ReadVariableOp_32l
4sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOp4sequential_4/lstm_8/lstm_cell_8/split/ReadVariableOp2p
6sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOp6sequential_4/lstm_8/lstm_cell_8/split_1/ReadVariableOp26
sequential_4/lstm_8/whilesequential_4/lstm_8/while:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
Ф~
	
while_body_163316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ђ
while/lstm_cell_8/mulMulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulІ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1І
while/lstm_cell_8/mul_2Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2І
while/lstm_cell_8/mul_3Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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
В
д
%sequential_4_lstm_8_while_body_160227D
@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counterJ
Fsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations)
%sequential_4_lstm_8_while_placeholder+
'sequential_4_lstm_8_while_placeholder_1+
'sequential_4_lstm_8_while_placeholder_2+
'sequential_4_lstm_8_while_placeholder_3C
?sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1_0
{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_4_lstm_8_while_lstm_cell_8_split_readvariableop_resource_0:	V
Gsequential_4_lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0:	R
?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0:	 &
"sequential_4_lstm_8_while_identity(
$sequential_4_lstm_8_while_identity_1(
$sequential_4_lstm_8_while_identity_2(
$sequential_4_lstm_8_while_identity_3(
$sequential_4_lstm_8_while_identity_4(
$sequential_4_lstm_8_while_identity_5A
=sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1}
ysequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensorV
Csequential_4_lstm_8_while_lstm_cell_8_split_readvariableop_resource:	T
Esequential_4_lstm_8_while_lstm_cell_8_split_1_readvariableop_resource:	P
=sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource:	 Ђ4sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOpЂ6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_1Ђ6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_2Ђ6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_3Ђ:sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOpЂ<sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpы
Ksequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2M
Ksequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeЫ
=sequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_8_while_placeholderTsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02?
=sequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItemХ
5sequential_4/lstm_8/while/lstm_cell_8/ones_like/ShapeShape'sequential_4_lstm_8_while_placeholder_2*
T0*
_output_shapes
:27
5sequential_4/lstm_8/while/lstm_cell_8/ones_like/ShapeГ
5sequential_4/lstm_8/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5sequential_4/lstm_8/while/lstm_cell_8/ones_like/Const
/sequential_4/lstm_8/while/lstm_cell_8/ones_likeFill>sequential_4/lstm_8/while/lstm_cell_8/ones_like/Shape:output:0>sequential_4/lstm_8/while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/ones_likeА
5sequential_4/lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_4/lstm_8/while/lstm_cell_8/split/split_dimџ
:sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOpReadVariableOpEsequential_4_lstm_8_while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02<
:sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOpП
+sequential_4/lstm_8/while/lstm_cell_8/splitSplit>sequential_4/lstm_8/while/lstm_cell_8/split/split_dim:output:0Bsequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2-
+sequential_4/lstm_8/while/lstm_cell_8/split
,sequential_4/lstm_8/while/lstm_cell_8/MatMulMatMulDsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_4/lstm_8/while/lstm_cell_8/MatMul
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_1MatMulDsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_1
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_2MatMulDsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_2
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_3MatMulDsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_3Д
7sequential_4/lstm_8/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_4/lstm_8/while/lstm_cell_8/split_1/split_dim
<sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOpGsequential_4_lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02>
<sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpЗ
-sequential_4/lstm_8/while/lstm_cell_8/split_1Split@sequential_4/lstm_8/while/lstm_cell_8/split_1/split_dim:output:0Dsequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2/
-sequential_4/lstm_8/while/lstm_cell_8/split_1
-sequential_4/lstm_8/while/lstm_cell_8/BiasAddBiasAdd6sequential_4/lstm_8/while/lstm_cell_8/MatMul:product:06sequential_4/lstm_8/while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_4/lstm_8/while/lstm_cell_8/BiasAdd
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_1BiasAdd8sequential_4/lstm_8/while/lstm_cell_8/MatMul_1:product:06sequential_4/lstm_8/while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_1
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_2BiasAdd8sequential_4/lstm_8/while/lstm_cell_8/MatMul_2:product:06sequential_4/lstm_8/while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_2
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_3BiasAdd8sequential_4/lstm_8/while/lstm_cell_8/MatMul_3:product:06sequential_4/lstm_8/while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_3ђ
)sequential_4/lstm_8/while/lstm_cell_8/mulMul'sequential_4_lstm_8_while_placeholder_28sequential_4/lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/while/lstm_cell_8/mulі
+sequential_4/lstm_8/while/lstm_cell_8/mul_1Mul'sequential_4_lstm_8_while_placeholder_28sequential_4/lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_1і
+sequential_4/lstm_8/while/lstm_cell_8/mul_2Mul'sequential_4_lstm_8_while_placeholder_28sequential_4/lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_2і
+sequential_4/lstm_8/while/lstm_cell_8/mul_3Mul'sequential_4_lstm_8_while_placeholder_28sequential_4/lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_3э
4sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOpReadVariableOp?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype026
4sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOpЧ
9sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stackЫ
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_1Ы
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_2р
3sequential_4/lstm_8/while/lstm_cell_8/strided_sliceStridedSlice<sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp:value:0Bsequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack:output:0Dsequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_1:output:0Dsequential_4/lstm_8/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask25
3sequential_4/lstm_8/while/lstm_cell_8/strided_slice
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_4MatMul-sequential_4/lstm_8/while/lstm_cell_8/mul:z:0<sequential_4/lstm_8/while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_4
)sequential_4/lstm_8/while/lstm_cell_8/addAddV26sequential_4/lstm_8/while/lstm_cell_8/BiasAdd:output:08sequential_4/lstm_8/while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_4/lstm_8/while/lstm_cell_8/addЪ
-sequential_4/lstm_8/while/lstm_cell_8/SigmoidSigmoid-sequential_4/lstm_8/while/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_4/lstm_8/while/lstm_cell_8/Sigmoidё
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_1Ы
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stackЯ
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Я
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_2ь
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1StridedSlice>sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_1:value:0Dsequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_1:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_5MatMul/sequential_4/lstm_8/while/lstm_cell_8/mul_1:z:0>sequential_4/lstm_8/while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_5
+sequential_4/lstm_8/while/lstm_cell_8/add_1AddV28sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_1:output:08sequential_4/lstm_8/while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/add_1а
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid/sequential_4/lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1ё
+sequential_4/lstm_8/while/lstm_cell_8/mul_4Mul3sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1:y:0'sequential_4_lstm_8_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_4ё
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_2Ы
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2=
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stackЯ
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Я
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_2ь
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2StridedSlice>sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_2:value:0Dsequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_1:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_6MatMul/sequential_4/lstm_8/while/lstm_cell_8/mul_2:z:0>sequential_4/lstm_8/while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_6
+sequential_4/lstm_8/while/lstm_cell_8/add_2AddV28sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_2:output:08sequential_4/lstm_8/while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/add_2У
*sequential_4/lstm_8/while/lstm_cell_8/ReluRelu/sequential_4/lstm_8/while/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_4/lstm_8/while/lstm_cell_8/Relu
+sequential_4/lstm_8/while/lstm_cell_8/mul_5Mul1sequential_4/lstm_8/while/lstm_cell_8/Sigmoid:y:08sequential_4/lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_5ї
+sequential_4/lstm_8/while/lstm_cell_8/add_3AddV2/sequential_4/lstm_8/while/lstm_cell_8/mul_4:z:0/sequential_4/lstm_8/while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/add_3ё
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_3Ы
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2=
;sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stackЯ
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Я
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_2ь
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3StridedSlice>sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_3:value:0Dsequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_1:output:0Fsequential_4/lstm_8/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_7MatMul/sequential_4/lstm_8/while/lstm_cell_8/mul_3:z:0>sequential_4/lstm_8/while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_7
+sequential_4/lstm_8/while/lstm_cell_8/add_4AddV28sequential_4/lstm_8/while/lstm_cell_8/BiasAdd_3:output:08sequential_4/lstm_8/while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/add_4а
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid/sequential_4/lstm_8/while/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2Ч
,sequential_4/lstm_8/while/lstm_cell_8/Relu_1Relu/sequential_4/lstm_8/while/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_4/lstm_8/while/lstm_cell_8/Relu_1
+sequential_4/lstm_8/while/lstm_cell_8/mul_6Mul3sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2:y:0:sequential_4/lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_6У
>sequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_8_while_placeholder_1%sequential_4_lstm_8_while_placeholder/sequential_4/lstm_8/while/lstm_cell_8/mul_6:z:0*
_output_shapes
: *
element_dtype02@
>sequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItem
sequential_4/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_4/lstm_8/while/add/yЙ
sequential_4/lstm_8/while/addAddV2%sequential_4_lstm_8_while_placeholder(sequential_4/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_8/while/add
!sequential_4/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_4/lstm_8/while/add_1/yк
sequential_4/lstm_8/while/add_1AddV2@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counter*sequential_4/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_8/while/add_1Л
"sequential_4/lstm_8/while/IdentityIdentity#sequential_4/lstm_8/while/add_1:z:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_4/lstm_8/while/Identityт
$sequential_4/lstm_8/while/Identity_1IdentityFsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_1Н
$sequential_4/lstm_8/while/Identity_2Identity!sequential_4/lstm_8/while/add:z:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_2ъ
$sequential_4/lstm_8/while/Identity_3IdentityNsequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_3м
$sequential_4/lstm_8/while/Identity_4Identity/sequential_4/lstm_8/while/lstm_cell_8/mul_6:z:0^sequential_4/lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_4/lstm_8/while/Identity_4м
$sequential_4/lstm_8/while/Identity_5Identity/sequential_4/lstm_8/while/lstm_cell_8/add_3:z:0^sequential_4/lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_4/lstm_8/while/Identity_5р
sequential_4/lstm_8/while/NoOpNoOp5^sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp7^sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_17^sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_27^sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_3;^sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOp=^sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_4/lstm_8/while/NoOp"Q
"sequential_4_lstm_8_while_identity+sequential_4/lstm_8/while/Identity:output:0"U
$sequential_4_lstm_8_while_identity_1-sequential_4/lstm_8/while/Identity_1:output:0"U
$sequential_4_lstm_8_while_identity_2-sequential_4/lstm_8/while/Identity_2:output:0"U
$sequential_4_lstm_8_while_identity_3-sequential_4/lstm_8/while/Identity_3:output:0"U
$sequential_4_lstm_8_while_identity_4-sequential_4/lstm_8/while/Identity_4:output:0"U
$sequential_4_lstm_8_while_identity_5-sequential_4/lstm_8/while/Identity_5:output:0"
=sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource?sequential_4_lstm_8_while_lstm_cell_8_readvariableop_resource_0"
Esequential_4_lstm_8_while_lstm_cell_8_split_1_readvariableop_resourceGsequential_4_lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0"
Csequential_4_lstm_8_while_lstm_cell_8_split_readvariableop_resourceEsequential_4_lstm_8_while_lstm_cell_8_split_readvariableop_resource_0"
=sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1?sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1_0"ј
ysequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2l
4sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp4sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp2p
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_16sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_12p
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_26sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_22p
6sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_36sequential_4/lstm_8/while/lstm_cell_8/ReadVariableOp_32x
:sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOp:sequential_4/lstm_8/while/lstm_cell_8/split/ReadVariableOp2|
<sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp<sequential_4/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp: 
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
v
ш
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_164057

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
 *UUе?2
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
dropout/Shapeв
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ъћ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Щмл2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЯѕБ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
 *UUе?2
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
dropout_3/Shapeи
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2у2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
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
:	*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6й
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ :џџџџџџџџџ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
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
е
У
while_cond_163590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163590___redundant_placeholder04
0while_while_cond_163590___redundant_placeholder14
0while_while_cond_163590___redundant_placeholder24
0while_while_cond_163590___redundant_placeholder3
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
Ш

lstm_8_while_body_162425*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0:	I
:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0:	E
2lstm_8_while_lstm_cell_8_readvariableop_resource_0:	 
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorI
6lstm_8_while_lstm_cell_8_split_readvariableop_resource:	G
8lstm_8_while_lstm_cell_8_split_1_readvariableop_resource:	C
0lstm_8_while_lstm_cell_8_readvariableop_resource:	 Ђ'lstm_8/while/lstm_cell_8/ReadVariableOpЂ)lstm_8/while/lstm_cell_8/ReadVariableOp_1Ђ)lstm_8/while/lstm_cell_8/ReadVariableOp_2Ђ)lstm_8/while/lstm_cell_8/ReadVariableOp_3Ђ-lstm_8/while/lstm_cell_8/split/ReadVariableOpЂ/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpб
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem
(lstm_8/while/lstm_cell_8/ones_like/ShapeShapelstm_8_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm_8/while/lstm_cell_8/ones_like/Shape
(lstm_8/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm_8/while/lstm_cell_8/ones_like/Constш
"lstm_8/while/lstm_cell_8/ones_likeFill1lstm_8/while/lstm_cell_8/ones_like/Shape:output:01lstm_8/while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/ones_like
&lstm_8/while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2(
&lstm_8/while/lstm_cell_8/dropout/Constу
$lstm_8/while/lstm_cell_8/dropout/MulMul+lstm_8/while/lstm_cell_8/ones_like:output:0/lstm_8/while/lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_8/while/lstm_cell_8/dropout/MulЋ
&lstm_8/while/lstm_cell_8/dropout/ShapeShape+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_8/while/lstm_cell_8/dropout/Shape
=lstm_8/while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform/lstm_8/while/lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2дду2?
=lstm_8/while/lstm_cell_8/dropout/random_uniform/RandomUniformЇ
/lstm_8/while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>21
/lstm_8/while/lstm_cell_8/dropout/GreaterEqual/yЂ
-lstm_8/while/lstm_cell_8/dropout/GreaterEqualGreaterEqualFlstm_8/while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:08lstm_8/while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-lstm_8/while/lstm_cell_8/dropout/GreaterEqualЪ
%lstm_8/while/lstm_cell_8/dropout/CastCast1lstm_8/while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_8/while/lstm_cell_8/dropout/Castо
&lstm_8/while/lstm_cell_8/dropout/Mul_1Mul(lstm_8/while/lstm_cell_8/dropout/Mul:z:0)lstm_8/while/lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_8/while/lstm_cell_8/dropout/Mul_1
(lstm_8/while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2*
(lstm_8/while/lstm_cell_8/dropout_1/Constщ
&lstm_8/while/lstm_cell_8/dropout_1/MulMul+lstm_8/while/lstm_cell_8/ones_like:output:01lstm_8/while/lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_8/while/lstm_cell_8/dropout_1/MulЏ
(lstm_8/while/lstm_cell_8/dropout_1/ShapeShape+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_8/while/lstm_cell_8/dropout_1/ShapeЄ
?lstm_8/while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_8/while/lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2рќџ2A
?lstm_8/while/lstm_cell_8/dropout_1/random_uniform/RandomUniformЋ
1lstm_8/while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>23
1lstm_8/while/lstm_cell_8/dropout_1/GreaterEqual/yЊ
/lstm_8/while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualHlstm_8/while/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0:lstm_8/while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_8/while/lstm_cell_8/dropout_1/GreaterEqualа
'lstm_8/while/lstm_cell_8/dropout_1/CastCast3lstm_8/while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_8/while/lstm_cell_8/dropout_1/Castц
(lstm_8/while/lstm_cell_8/dropout_1/Mul_1Mul*lstm_8/while/lstm_cell_8/dropout_1/Mul:z:0+lstm_8/while/lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_8/while/lstm_cell_8/dropout_1/Mul_1
(lstm_8/while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2*
(lstm_8/while/lstm_cell_8/dropout_2/Constщ
&lstm_8/while/lstm_cell_8/dropout_2/MulMul+lstm_8/while/lstm_cell_8/ones_like:output:01lstm_8/while/lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_8/while/lstm_cell_8/dropout_2/MulЏ
(lstm_8/while/lstm_cell_8/dropout_2/ShapeShape+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_8/while/lstm_cell_8/dropout_2/ShapeЃ
?lstm_8/while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_8/while/lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2u2A
?lstm_8/while/lstm_cell_8/dropout_2/random_uniform/RandomUniformЋ
1lstm_8/while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>23
1lstm_8/while/lstm_cell_8/dropout_2/GreaterEqual/yЊ
/lstm_8/while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualHlstm_8/while/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0:lstm_8/while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_8/while/lstm_cell_8/dropout_2/GreaterEqualа
'lstm_8/while/lstm_cell_8/dropout_2/CastCast3lstm_8/while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_8/while/lstm_cell_8/dropout_2/Castц
(lstm_8/while/lstm_cell_8/dropout_2/Mul_1Mul*lstm_8/while/lstm_cell_8/dropout_2/Mul:z:0+lstm_8/while/lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_8/while/lstm_cell_8/dropout_2/Mul_1
(lstm_8/while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2*
(lstm_8/while/lstm_cell_8/dropout_3/Constщ
&lstm_8/while/lstm_cell_8/dropout_3/MulMul+lstm_8/while/lstm_cell_8/ones_like:output:01lstm_8/while/lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_8/while/lstm_cell_8/dropout_3/MulЏ
(lstm_8/while/lstm_cell_8/dropout_3/ShapeShape+lstm_8/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_8/while/lstm_cell_8/dropout_3/ShapeЄ
?lstm_8/while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_8/while/lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Њая2A
?lstm_8/while/lstm_cell_8/dropout_3/random_uniform/RandomUniformЋ
1lstm_8/while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>23
1lstm_8/while/lstm_cell_8/dropout_3/GreaterEqual/yЊ
/lstm_8/while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualHlstm_8/while/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0:lstm_8/while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_8/while/lstm_cell_8/dropout_3/GreaterEqualа
'lstm_8/while/lstm_cell_8/dropout_3/CastCast3lstm_8/while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_8/while/lstm_cell_8/dropout_3/Castц
(lstm_8/while/lstm_cell_8/dropout_3/Mul_1Mul*lstm_8/while/lstm_cell_8/dropout_3/Mul:z:0+lstm_8/while/lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_8/while/lstm_cell_8/dropout_3/Mul_1
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dimи
-lstm_8/while/lstm_cell_8/split/ReadVariableOpReadVariableOp8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02/
-lstm_8/while/lstm_cell_8/split/ReadVariableOp
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:05lstm_8/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2 
lstm_8/while/lstm_cell_8/splitр
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_8/while/lstm_cell_8/MatMulф
!lstm_8/while/lstm_cell_8/MatMul_1MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_1ф
!lstm_8/while/lstm_cell_8/MatMul_2MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_2ф
!lstm_8/while/lstm_cell_8/MatMul_3MatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_3
*lstm_8/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_8/while/lstm_cell_8/split_1/split_dimк
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp
 lstm_8/while/lstm_cell_8/split_1Split3lstm_8/while/lstm_cell_8/split_1/split_dim:output:07lstm_8/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2"
 lstm_8/while/lstm_cell_8/split_1з
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd)lstm_8/while/lstm_cell_8/MatMul:product:0)lstm_8/while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/while/lstm_cell_8/BiasAddн
"lstm_8/while/lstm_cell_8/BiasAdd_1BiasAdd+lstm_8/while/lstm_cell_8/MatMul_1:product:0)lstm_8/while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_1н
"lstm_8/while/lstm_cell_8/BiasAdd_2BiasAdd+lstm_8/while/lstm_cell_8/MatMul_2:product:0)lstm_8/while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_2н
"lstm_8/while/lstm_cell_8/BiasAdd_3BiasAdd+lstm_8/while/lstm_cell_8/MatMul_3:product:0)lstm_8/while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/BiasAdd_3Н
lstm_8/while/lstm_cell_8/mulMullstm_8_while_placeholder_2*lstm_8/while/lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/mulУ
lstm_8/while/lstm_cell_8/mul_1Mullstm_8_while_placeholder_2,lstm_8/while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_1У
lstm_8/while/lstm_cell_8/mul_2Mullstm_8_while_placeholder_2,lstm_8/while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_2У
lstm_8/while/lstm_cell_8/mul_3Mullstm_8_while_placeholder_2,lstm_8/while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_3Ц
'lstm_8/while/lstm_cell_8/ReadVariableOpReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02)
'lstm_8/while/lstm_cell_8/ReadVariableOp­
,lstm_8/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_8/while/lstm_cell_8/strided_slice/stackБ
.lstm_8/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_8/while/lstm_cell_8/strided_slice/stack_1Б
.lstm_8/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_8/while/lstm_cell_8/strided_slice/stack_2
&lstm_8/while/lstm_cell_8/strided_sliceStridedSlice/lstm_8/while/lstm_cell_8/ReadVariableOp:value:05lstm_8/while/lstm_cell_8/strided_slice/stack:output:07lstm_8/while/lstm_cell_8/strided_slice/stack_1:output:07lstm_8/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_8/while/lstm_cell_8/strided_sliceе
!lstm_8/while/lstm_cell_8/MatMul_4MatMul lstm_8/while/lstm_cell_8/mul:z:0/lstm_8/while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_4Я
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/BiasAdd:output:0+lstm_8/while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/addЃ
 lstm_8/while/lstm_cell_8/SigmoidSigmoid lstm_8/while/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/while/lstm_cell_8/SigmoidЪ
)lstm_8/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_1Б
.lstm_8/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_8/while/lstm_cell_8/strided_slice_1/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_1/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_1StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_1:value:07lstm_8/while/lstm_cell_8/strided_slice_1/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_1/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_1й
!lstm_8/while/lstm_cell_8/MatMul_5MatMul"lstm_8/while/lstm_cell_8/mul_1:z:01lstm_8/while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_5е
lstm_8/while/lstm_cell_8/add_1AddV2+lstm_8/while/lstm_cell_8/BiasAdd_1:output:0+lstm_8/while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_1Љ
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/Sigmoid_1Н
lstm_8/while/lstm_cell_8/mul_4Mul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_4Ъ
)lstm_8/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_2Б
.lstm_8/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_8/while/lstm_cell_8/strided_slice_2/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_2/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_2StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_2:value:07lstm_8/while/lstm_cell_8/strided_slice_2/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_2/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_2й
!lstm_8/while/lstm_cell_8/MatMul_6MatMul"lstm_8/while/lstm_cell_8/mul_2:z:01lstm_8/while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_6е
lstm_8/while/lstm_cell_8/add_2AddV2+lstm_8/while/lstm_cell_8/BiasAdd_2:output:0+lstm_8/while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_2
lstm_8/while/lstm_cell_8/ReluRelu"lstm_8/while/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/lstm_cell_8/ReluЬ
lstm_8/while/lstm_cell_8/mul_5Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_5У
lstm_8/while/lstm_cell_8/add_3AddV2"lstm_8/while/lstm_cell_8/mul_4:z:0"lstm_8/while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_3Ъ
)lstm_8/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp2lstm_8_while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_8/while/lstm_cell_8/ReadVariableOp_3Б
.lstm_8/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_8/while/lstm_cell_8/strided_slice_3/stackЕ
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_1Е
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_8/while/lstm_cell_8/strided_slice_3/stack_2
(lstm_8/while/lstm_cell_8/strided_slice_3StridedSlice1lstm_8/while/lstm_cell_8/ReadVariableOp_3:value:07lstm_8/while/lstm_cell_8/strided_slice_3/stack:output:09lstm_8/while/lstm_cell_8/strided_slice_3/stack_1:output:09lstm_8/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_8/while/lstm_cell_8/strided_slice_3й
!lstm_8/while/lstm_cell_8/MatMul_7MatMul"lstm_8/while/lstm_cell_8/mul_3:z:01lstm_8/while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/while/lstm_cell_8/MatMul_7е
lstm_8/while/lstm_cell_8/add_4AddV2+lstm_8/while/lstm_cell_8/BiasAdd_3:output:0+lstm_8/while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/add_4Љ
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid"lstm_8/while/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/while/lstm_cell_8/Sigmoid_2 
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_8/while/lstm_cell_8/Relu_1а
lstm_8/while/lstm_cell_8/mul_6Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/while/lstm_cell_8/mul_6
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_6:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/IdentityЁ
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2Ж
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3Ј
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_6:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/Identity_4Ј
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_3:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/while/Identity_5ј
lstm_8/while/NoOpNoOp(^lstm_8/while/lstm_cell_8/ReadVariableOp*^lstm_8/while/lstm_cell_8/ReadVariableOp_1*^lstm_8/while/lstm_cell_8/ReadVariableOp_2*^lstm_8/while/lstm_cell_8/ReadVariableOp_3.^lstm_8/while/lstm_cell_8/split/ReadVariableOp0^lstm_8/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_8/while/NoOp"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"f
0lstm_8_while_lstm_cell_8_readvariableop_resource2lstm_8_while_lstm_cell_8_readvariableop_resource_0"v
8lstm_8_while_lstm_cell_8_split_1_readvariableop_resource:lstm_8_while_lstm_cell_8_split_1_readvariableop_resource_0"r
6lstm_8_while_lstm_cell_8_split_readvariableop_resource8lstm_8_while_lstm_cell_8_split_readvariableop_resource_0"Ф
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2R
'lstm_8/while/lstm_cell_8/ReadVariableOp'lstm_8/while/lstm_cell_8/ReadVariableOp2V
)lstm_8/while/lstm_cell_8/ReadVariableOp_1)lstm_8/while/lstm_cell_8/ReadVariableOp_12V
)lstm_8/while/lstm_cell_8/ReadVariableOp_2)lstm_8/while/lstm_cell_8/ReadVariableOp_22V
)lstm_8/while/lstm_cell_8/ReadVariableOp_3)lstm_8/while/lstm_cell_8/ReadVariableOp_32^
-lstm_8/while/lstm_cell_8/split/ReadVariableOp-lstm_8/while/lstm_cell_8/split/ReadVariableOp2b
/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp/lstm_8/while/lstm_cell_8/split_1/ReadVariableOp: 
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
ч
г
H__inference_sequential_4_layer_call_and_return_conditional_losses_162612

inputsC
0lstm_8_lstm_cell_8_split_readvariableop_resource:	A
2lstm_8_lstm_cell_8_split_1_readvariableop_resource:	=
*lstm_8_lstm_cell_8_readvariableop_resource:	 9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource: 6
(dense_13_biasadd_readvariableop_resource:
identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂ!lstm_8/lstm_cell_8/ReadVariableOpЂ#lstm_8/lstm_cell_8/ReadVariableOp_1Ђ#lstm_8/lstm_cell_8/ReadVariableOp_2Ђ#lstm_8/lstm_cell_8/ReadVariableOp_3Ђ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂ'lstm_8/lstm_cell_8/split/ReadVariableOpЂ)lstm_8/lstm_cell_8/split_1/ReadVariableOpЂlstm_8/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/Shape
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros/mul/y
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_8/zeros/Less/y
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros/packed/1
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros_1/mul/y
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_8/zeros_1/Less/y
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/zeros_1/packed/1Ѕ
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/zeros_1
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_8/TensorArrayV2/element_shapeЮ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2Э
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2І
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_8/strided_slice_2
"lstm_8/lstm_cell_8/ones_like/ShapeShapelstm_8/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_8/lstm_cell_8/ones_like/Shape
"lstm_8/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm_8/lstm_cell_8/ones_like/Constа
lstm_8/lstm_cell_8/ones_likeFill+lstm_8/lstm_cell_8/ones_like/Shape:output:0+lstm_8/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/ones_like
 lstm_8/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2"
 lstm_8/lstm_cell_8/dropout/ConstЫ
lstm_8/lstm_cell_8/dropout/MulMul%lstm_8/lstm_cell_8/ones_like:output:0)lstm_8/lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_8/lstm_cell_8/dropout/Mul
 lstm_8/lstm_cell_8/dropout/ShapeShape%lstm_8/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_8/lstm_cell_8/dropout/Shape
7lstm_8/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform)lstm_8/lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Экh29
7lstm_8/lstm_cell_8/dropout/random_uniform/RandomUniform
)lstm_8/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2+
)lstm_8/lstm_cell_8/dropout/GreaterEqual/y
'lstm_8/lstm_cell_8/dropout/GreaterEqualGreaterEqual@lstm_8/lstm_cell_8/dropout/random_uniform/RandomUniform:output:02lstm_8/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_8/lstm_cell_8/dropout/GreaterEqualИ
lstm_8/lstm_cell_8/dropout/CastCast+lstm_8/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
lstm_8/lstm_cell_8/dropout/CastЦ
 lstm_8/lstm_cell_8/dropout/Mul_1Mul"lstm_8/lstm_cell_8/dropout/Mul:z:0#lstm_8/lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/lstm_cell_8/dropout/Mul_1
"lstm_8/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2$
"lstm_8/lstm_cell_8/dropout_1/Constб
 lstm_8/lstm_cell_8/dropout_1/MulMul%lstm_8/lstm_cell_8/ones_like:output:0+lstm_8/lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/lstm_cell_8/dropout_1/Mul
"lstm_8/lstm_cell_8/dropout_1/ShapeShape%lstm_8/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_8/lstm_cell_8/dropout_1/Shape
9lstm_8/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_8/lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ћЬ2;
9lstm_8/lstm_cell_8/dropout_1/random_uniform/RandomUniform
+lstm_8/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2-
+lstm_8/lstm_cell_8/dropout_1/GreaterEqual/y
)lstm_8/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualBlstm_8/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:04lstm_8/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_8/lstm_cell_8/dropout_1/GreaterEqualО
!lstm_8/lstm_cell_8/dropout_1/CastCast-lstm_8/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/lstm_cell_8/dropout_1/CastЮ
"lstm_8/lstm_cell_8/dropout_1/Mul_1Mul$lstm_8/lstm_cell_8/dropout_1/Mul:z:0%lstm_8/lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/lstm_cell_8/dropout_1/Mul_1
"lstm_8/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2$
"lstm_8/lstm_cell_8/dropout_2/Constб
 lstm_8/lstm_cell_8/dropout_2/MulMul%lstm_8/lstm_cell_8/ones_like:output:0+lstm_8/lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/lstm_cell_8/dropout_2/Mul
"lstm_8/lstm_cell_8/dropout_2/ShapeShape%lstm_8/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_8/lstm_cell_8/dropout_2/Shape
9lstm_8/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_8/lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2њ Ў2;
9lstm_8/lstm_cell_8/dropout_2/random_uniform/RandomUniform
+lstm_8/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2-
+lstm_8/lstm_cell_8/dropout_2/GreaterEqual/y
)lstm_8/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualBlstm_8/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:04lstm_8/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_8/lstm_cell_8/dropout_2/GreaterEqualО
!lstm_8/lstm_cell_8/dropout_2/CastCast-lstm_8/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/lstm_cell_8/dropout_2/CastЮ
"lstm_8/lstm_cell_8/dropout_2/Mul_1Mul$lstm_8/lstm_cell_8/dropout_2/Mul:z:0%lstm_8/lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/lstm_cell_8/dropout_2/Mul_1
"lstm_8/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2$
"lstm_8/lstm_cell_8/dropout_3/Constб
 lstm_8/lstm_cell_8/dropout_3/MulMul%lstm_8/lstm_cell_8/ones_like:output:0+lstm_8/lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_8/lstm_cell_8/dropout_3/Mul
"lstm_8/lstm_cell_8/dropout_3/ShapeShape%lstm_8/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_8/lstm_cell_8/dropout_3/Shape
9lstm_8/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_8/lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2дАф2;
9lstm_8/lstm_cell_8/dropout_3/random_uniform/RandomUniform
+lstm_8/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2-
+lstm_8/lstm_cell_8/dropout_3/GreaterEqual/y
)lstm_8/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualBlstm_8/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:04lstm_8/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_8/lstm_cell_8/dropout_3/GreaterEqualО
!lstm_8/lstm_cell_8/dropout_3/CastCast-lstm_8/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_8/lstm_cell_8/dropout_3/CastЮ
"lstm_8/lstm_cell_8/dropout_3/Mul_1Mul$lstm_8/lstm_cell_8/dropout_3/Mul:z:0%lstm_8/lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_8/lstm_cell_8/dropout_3/Mul_1
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dimФ
'lstm_8/lstm_cell_8/split/ReadVariableOpReadVariableOp0lstm_8_lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02)
'lstm_8/lstm_cell_8/split/ReadVariableOpѓ
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0/lstm_8/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_8/lstm_cell_8/splitЖ
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMulК
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_1К
lstm_8/lstm_cell_8/MatMul_2MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_2К
lstm_8/lstm_cell_8/MatMul_3MatMullstm_8/strided_slice_2:output:0!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_3
$lstm_8/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_8/lstm_cell_8/split_1/split_dimЦ
)lstm_8/lstm_cell_8/split_1/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_8/lstm_cell_8/split_1/ReadVariableOpы
lstm_8/lstm_cell_8/split_1Split-lstm_8/lstm_cell_8/split_1/split_dim:output:01lstm_8/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_8/lstm_cell_8/split_1П
lstm_8/lstm_cell_8/BiasAddBiasAdd#lstm_8/lstm_cell_8/MatMul:product:0#lstm_8/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAddХ
lstm_8/lstm_cell_8/BiasAdd_1BiasAdd%lstm_8/lstm_cell_8/MatMul_1:product:0#lstm_8/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_1Х
lstm_8/lstm_cell_8/BiasAdd_2BiasAdd%lstm_8/lstm_cell_8/MatMul_2:product:0#lstm_8/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_2Х
lstm_8/lstm_cell_8/BiasAdd_3BiasAdd%lstm_8/lstm_cell_8/MatMul_3:product:0#lstm_8/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/BiasAdd_3І
lstm_8/lstm_cell_8/mulMullstm_8/zeros:output:0$lstm_8/lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mulЌ
lstm_8/lstm_cell_8/mul_1Mullstm_8/zeros:output:0&lstm_8/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_1Ќ
lstm_8/lstm_cell_8/mul_2Mullstm_8/zeros:output:0&lstm_8/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_2Ќ
lstm_8/lstm_cell_8/mul_3Mullstm_8/zeros:output:0&lstm_8/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_3В
!lstm_8/lstm_cell_8/ReadVariableOpReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02#
!lstm_8/lstm_cell_8/ReadVariableOpЁ
&lstm_8/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_8/lstm_cell_8/strided_slice/stackЅ
(lstm_8/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_8/lstm_cell_8/strided_slice/stack_1Ѕ
(lstm_8/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_8/lstm_cell_8/strided_slice/stack_2ю
 lstm_8/lstm_cell_8/strided_sliceStridedSlice)lstm_8/lstm_cell_8/ReadVariableOp:value:0/lstm_8/lstm_cell_8/strided_slice/stack:output:01lstm_8/lstm_cell_8/strided_slice/stack_1:output:01lstm_8/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 lstm_8/lstm_cell_8/strided_sliceН
lstm_8/lstm_cell_8/MatMul_4MatMullstm_8/lstm_cell_8/mul:z:0)lstm_8/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_4З
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/BiasAdd:output:0%lstm_8/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add
lstm_8/lstm_cell_8/SigmoidSigmoidlstm_8/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/SigmoidЖ
#lstm_8/lstm_cell_8/ReadVariableOp_1ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_1Ѕ
(lstm_8/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_8/lstm_cell_8/strided_slice_1/stackЉ
*lstm_8/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_8/lstm_cell_8/strided_slice_1/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_1/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_1StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_1:value:01lstm_8/lstm_cell_8/strided_slice_1/stack:output:03lstm_8/lstm_cell_8/strided_slice_1/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_1С
lstm_8/lstm_cell_8/MatMul_5MatMullstm_8/lstm_cell_8/mul_1:z:0+lstm_8/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_5Н
lstm_8/lstm_cell_8/add_1AddV2%lstm_8/lstm_cell_8/BiasAdd_1:output:0%lstm_8/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_1
lstm_8/lstm_cell_8/Sigmoid_1Sigmoidlstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Sigmoid_1Ј
lstm_8/lstm_cell_8/mul_4Mul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_4Ж
#lstm_8/lstm_cell_8/ReadVariableOp_2ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_2Ѕ
(lstm_8/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_8/lstm_cell_8/strided_slice_2/stackЉ
*lstm_8/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_8/lstm_cell_8/strided_slice_2/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_2/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_2StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_2:value:01lstm_8/lstm_cell_8/strided_slice_2/stack:output:03lstm_8/lstm_cell_8/strided_slice_2/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_2С
lstm_8/lstm_cell_8/MatMul_6MatMullstm_8/lstm_cell_8/mul_2:z:0+lstm_8/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_6Н
lstm_8/lstm_cell_8/add_2AddV2%lstm_8/lstm_cell_8/BiasAdd_2:output:0%lstm_8/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_2
lstm_8/lstm_cell_8/ReluRelulstm_8/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/ReluД
lstm_8/lstm_cell_8/mul_5Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_5Ћ
lstm_8/lstm_cell_8/add_3AddV2lstm_8/lstm_cell_8/mul_4:z:0lstm_8/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_3Ж
#lstm_8/lstm_cell_8/ReadVariableOp_3ReadVariableOp*lstm_8_lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_8/lstm_cell_8/ReadVariableOp_3Ѕ
(lstm_8/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(lstm_8/lstm_cell_8/strided_slice_3/stackЉ
*lstm_8/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_8/lstm_cell_8/strided_slice_3/stack_1Љ
*lstm_8/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_8/lstm_cell_8/strided_slice_3/stack_2њ
"lstm_8/lstm_cell_8/strided_slice_3StridedSlice+lstm_8/lstm_cell_8/ReadVariableOp_3:value:01lstm_8/lstm_cell_8/strided_slice_3/stack:output:03lstm_8/lstm_cell_8/strided_slice_3/stack_1:output:03lstm_8/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_8/lstm_cell_8/strided_slice_3С
lstm_8/lstm_cell_8/MatMul_7MatMullstm_8/lstm_cell_8/mul_3:z:0+lstm_8/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/MatMul_7Н
lstm_8/lstm_cell_8/add_4AddV2%lstm_8/lstm_cell_8/BiasAdd_3:output:0%lstm_8/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/add_4
lstm_8/lstm_cell_8/Sigmoid_2Sigmoidlstm_8/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Sigmoid_2
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/Relu_1И
lstm_8/lstm_cell_8/mul_6Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_8/lstm_cell_8/mul_6
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2&
$lstm_8/TensorArrayV2_1/element_shapeд
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterч
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_8_lstm_cell_8_split_readvariableop_resource2lstm_8_lstm_cell_8_split_1_readvariableop_resource*lstm_8_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_8_while_body_162425*$
condR
lstm_8_while_cond_162424*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_8/whileУ
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_8/strided_slice_3/stack
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2Ф
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_8/strided_slice_3
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/permС
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimeЈ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_12/MatMul/ReadVariableOpЇ
dense_12/MatMulMatMullstm_8/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/MatMulЇ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_12/BiasAdd/ReadVariableOpЅ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_12/ReluЈ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_13/MatMul/ReadVariableOpЃ
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_13/MatMulЇ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЅ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_13/BiasAddk
reshape_6/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_6/Shape
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stack
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2
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
reshape_6/Reshape/shape/2в
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shapeЄ
reshape_6/ReshapeReshapedense_13/BiasAdd:output:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_6/Reshapeь
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_8_lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muly
IdentityIdentityreshape_6/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp"^lstm_8/lstm_cell_8/ReadVariableOp$^lstm_8/lstm_cell_8/ReadVariableOp_1$^lstm_8/lstm_cell_8/ReadVariableOp_2$^lstm_8/lstm_cell_8/ReadVariableOp_3<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp(^lstm_8/lstm_cell_8/split/ReadVariableOp*^lstm_8/lstm_cell_8/split_1/ReadVariableOp^lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2F
!lstm_8/lstm_cell_8/ReadVariableOp!lstm_8/lstm_cell_8/ReadVariableOp2J
#lstm_8/lstm_cell_8/ReadVariableOp_1#lstm_8/lstm_cell_8/ReadVariableOp_12J
#lstm_8/lstm_cell_8/ReadVariableOp_2#lstm_8/lstm_cell_8/ReadVariableOp_22J
#lstm_8/lstm_cell_8/ReadVariableOp_3#lstm_8/lstm_cell_8/ReadVariableOp_32z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp2R
'lstm_8/lstm_cell_8/split/ReadVariableOp'lstm_8/lstm_cell_8/split/ReadVariableOp2V
)lstm_8/lstm_cell_8/split_1/ReadVariableOp)lstm_8/lstm_cell_8/split_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э!
ѓ
H__inference_sequential_4_layer_call_and_return_conditional_losses_161985
input_5 
lstm_8_161960:	
lstm_8_161962:	 
lstm_8_161964:	 !
dense_12_161967:  
dense_12_161969: !
dense_13_161972: 
dense_13_161974:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂlstm_8/StatefulPartitionedCallЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_8_161960lstm_8_161962lstm_8_161964*
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1618352 
lstm_8/StatefulPartitionedCallЕ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_12_161967dense_12_161969*
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
GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1614282"
 dense_12/StatefulPartitionedCallЗ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_161972dense_13_161974*
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
GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1614442"
 dense_13/StatefulPartitionedCallў
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_1614632
reshape_6/PartitionedCallЩ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_8_161960*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/mul
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityѓ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5
з

B__inference_lstm_8_layer_call_and_return_conditional_losses_163449

inputs<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileD
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
:џџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_163316*
condR
while_cond_163315*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
У
while_cond_161669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_161669___redundant_placeholder04
0while_while_cond_161669___redundant_placeholder14
0while_while_cond_161669___redundant_placeholder24
0while_while_cond_161669___redundant_placeholder3
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
е
У
while_cond_162765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_162765___redundant_placeholder04
0while_while_cond_162765___redundant_placeholder14
0while_while_cond_162765___redundant_placeholder24
0while_while_cond_162765___redundant_placeholder3
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
Ф~
	
while_body_162766
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ђ
while/lstm_cell_8/mulMulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulІ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1І
while/lstm_cell_8/mul_2Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2І
while/lstm_cell_8/mul_3Mulwhile_placeholder_2$while/lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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
 

B__inference_lstm_8_layer_call_and_return_conditional_losses_162899
inputs_0<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileF
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
 :џџџџџџџџџџџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_162766*
condR
while_cond_162765*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
МА
	
while_body_161670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_8_split_readvariableop_resource_0:	B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	>
+while_lstm_cell_8_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_8_split_readvariableop_resource:	@
1while_lstm_cell_8_split_1_readvariableop_resource:	<
)while_lstm_cell_8_readvariableop_resource:	 Ђ while/lstm_cell_8/ReadVariableOpЂ"while/lstm_cell_8/ReadVariableOp_1Ђ"while/lstm_cell_8/ReadVariableOp_2Ђ"while/lstm_cell_8/ReadVariableOp_3Ђ&while/lstm_cell_8/split/ReadVariableOpЂ(while/lstm_cell_8/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_8/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_8/ones_like/Shape
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_8/ones_like/ConstЬ
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ones_like
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2!
while/lstm_cell_8/dropout/ConstЧ
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/dropout/Mul
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_8/dropout/Shape
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ъш28
6while/lstm_cell_8/dropout/random_uniform/RandomUniform
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2*
(while/lstm_cell_8/dropout/GreaterEqual/y
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&while/lstm_cell_8/dropout/GreaterEqualЕ
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_8/dropout/CastТ
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout/Mul_1
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_1/ConstЭ
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_1/Mul
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_1/Shape
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ФЬ2:
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_1/GreaterEqual/y
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_1/GreaterEqualЛ
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_1/CastЪ
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_1/Mul_1
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_2/ConstЭ
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_2/Mul
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_2/Shape
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2йЪР2:
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_2/GreaterEqual/y
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_2/GreaterEqualЛ
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_2/CastЪ
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_2/Mul_1
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2#
!while/lstm_cell_8/dropout_3/ConstЭ
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_8/dropout_3/Mul
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_8/dropout_3/Shape
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЂСЗ2:
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2,
*while/lstm_cell_8/dropout_3/GreaterEqual/y
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_8/dropout_3/GreaterEqualЛ
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_8/dropout_3/CastЪ
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_8/dropout_3/Mul_1
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimУ
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/lstm_cell_8/split/ReadVariableOpя
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_8/splitФ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMulШ
while/lstm_cell_8/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_1Ш
while/lstm_cell_8/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_2Ш
while/lstm_cell_8/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_3
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_8/split_1/split_dimХ
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/split_1/ReadVariableOpч
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_8/split_1Л
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAddС
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_1С
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_2С
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/BiasAdd_3Ё
while/lstm_cell_8/mulMulwhile_placeholder_2#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mulЇ
while/lstm_cell_8/mul_1Mulwhile_placeholder_2%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_1Ї
while/lstm_cell_8/mul_2Mulwhile_placeholder_2%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_2Ї
while/lstm_cell_8/mul_3Mulwhile_placeholder_2%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_3Б
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02"
 while/lstm_cell_8/ReadVariableOp
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_8/strided_slice/stackЃ
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice/stack_1Ѓ
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_8/strided_slice/stack_2ш
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_8/strided_sliceЙ
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_4Г
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add
while/lstm_cell_8/SigmoidSigmoidwhile/lstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/SigmoidЕ
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_1Ѓ
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_8/strided_slice_1/stackЇ
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_8/strided_slice_1/stack_1Ї
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_1/stack_2є
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_1Н
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_1:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_5Й
while/lstm_cell_8/add_1AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_1Sigmoidwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_1Ё
while/lstm_cell_8/mul_4Mulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_4Е
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_2Ѓ
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_8/strided_slice_2/stackЇ
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_8/strided_slice_2/stack_1Ї
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_2/stack_2є
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_2Н
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_2:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_6Й
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_2
while/lstm_cell_8/ReluReluwhile/lstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_5Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_5Ї
while/lstm_cell_8/add_3AddV2while/lstm_cell_8/mul_4:z:0while/lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_3Е
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_8/ReadVariableOp_3Ѓ
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_8/strided_slice_3/stackЇ
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_8/strided_slice_3/stack_1Ї
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_8/strided_slice_3/stack_2є
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_8/strided_slice_3Н
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_3:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/MatMul_7Й
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/add_4
while/lstm_cell_8/Sigmoid_2Sigmoidwhile/lstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_6Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_8/mul_6п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Р

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 
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
е
У
while_cond_160810
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_160810___redundant_placeholder04
0while_while_cond_160810___redundant_placeholder14
0while_while_cond_160810___redundant_placeholder24
0while_while_cond_160810___redundant_placeholder3
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
І

ѕ
D__inference_dense_13_layer_call_and_return_conditional_losses_161444

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

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
ђЬ

B__inference_lstm_8_layer_call_and_return_conditional_losses_163756

inputs<
)lstm_cell_8_split_readvariableop_resource:	:
+lstm_cell_8_split_1_readvariableop_resource:	6
#lstm_cell_8_readvariableop_resource:	 
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_8/ReadVariableOpЂlstm_cell_8/ReadVariableOp_1Ђlstm_cell_8/ReadVariableOp_2Ђlstm_cell_8/ReadVariableOp_3Ђ lstm_cell_8/split/ReadVariableOpЂ"lstm_cell_8/split_1/ReadVariableOpЂwhileD
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
:џџџџџџџџџ2
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
valueB"џџџџ   27
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
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_8/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_8/ones_like/Shape
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_8/ones_like/ConstД
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/ones_like{
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout/ConstЏ
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout/Shapeї
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ђнх22
0lstm_cell_8/dropout/random_uniform/RandomUniform
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2$
"lstm_cell_8/dropout/GreaterEqual/yю
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_cell_8/dropout/GreaterEqualЃ
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/CastЊ
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout/Mul_1
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_1/ConstЕ
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_1/Shape§
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЬЄ24
2lstm_cell_8/dropout_1/random_uniform/RandomUniform
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_1/GreaterEqual/yі
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_1/GreaterEqualЉ
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/CastВ
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_1/Mul_1
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_2/ConstЕ
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_2/Shapeќ
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЋИ[24
2lstm_cell_8/dropout_2/random_uniform/RandomUniform
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_2/GreaterEqual/yі
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_2/GreaterEqualЉ
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/CastВ
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_2/Mul_1
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
lstm_cell_8/dropout_3/ConstЕ
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_8/dropout_3/Shape§
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2СМ24
2lstm_cell_8/dropout_3/random_uniform/RandomUniform
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2&
$lstm_cell_8/dropout_3/GreaterEqual/yі
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_8/dropout_3/GreaterEqualЉ
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/CastВ
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/dropout_3/Mul_1|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimЏ
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02"
 lstm_cell_8/split/ReadVariableOpз
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_8/split
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul
lstm_cell_8/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_1
lstm_cell_8/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_2
lstm_cell_8/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_3
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_8/split_1/split_dimБ
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/split_1/ReadVariableOpЯ
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_8/split_1Ѓ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAddЉ
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_1Љ
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_2Љ
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/BiasAdd_3
lstm_cell_8/mulMulzeros:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul
lstm_cell_8/mul_1Mulzeros:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_1
lstm_cell_8/mul_2Mulzeros:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_2
lstm_cell_8/mul_3Mulzeros:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_3
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_8/strided_slice/stack
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice/stack_1
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_8/strided_slice/stack_2Ф
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_sliceЁ
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul:z:0"lstm_cell_8/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_4
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add|
lstm_cell_8/SigmoidSigmoidlstm_cell_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/SigmoidЁ
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_1
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_8/strided_slice_1/stack
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_8/strided_slice_1/stack_1
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_1/stack_2а
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_1Ѕ
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_1:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_5Ё
lstm_cell_8/add_1AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mul_4Mullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_4Ё
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_2
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_8/strided_slice_2/stack
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_8/strided_slice_2/stack_1
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_2/stack_2а
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_2Ѕ
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_2:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_6Ё
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_2u
lstm_cell_8/ReluRelulstm_cell_8/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu
lstm_cell_8/mul_5Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_5
lstm_cell_8/add_3AddV2lstm_cell_8/mul_4:z:0lstm_cell_8/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_3Ё
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_8/ReadVariableOp_3
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_8/strided_slice_3/stack
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_8/strided_slice_3/stack_1
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_8/strided_slice_3/stack_2а
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_8/strided_slice_3Ѕ
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_3:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/MatMul_7Ё
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/add_4
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/Relu_1
lstm_cell_8/mul_6Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_8/mul_6
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
while/loop_counterў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
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
bodyR
while_body_163591*
condR
while_cond_163590*K
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
runtimeх
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityж
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ!
ђ
H__inference_sequential_4_layer_call_and_return_conditional_losses_161893

inputs 
lstm_8_161868:	
lstm_8_161870:	 
lstm_8_161872:	 !
dense_12_161875:  
dense_12_161877: !
dense_13_161880: 
dense_13_161882:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂlstm_8/StatefulPartitionedCallЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_161868lstm_8_161870lstm_8_161872*
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1618352 
lstm_8/StatefulPartitionedCallЕ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_12_161875dense_12_161877*
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
GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1614282"
 dense_12/StatefulPartitionedCallЗ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_161880dense_13_161882*
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
GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1614442"
 dense_13/StatefulPartitionedCallў
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_1614632
reshape_6/PartitionedCallЩ
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_8_161868*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/mul
IdentityIdentity"reshape_6/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityѓ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ
Ц
__inference_loss_fn_0_164102W
Dlstm_8_lstm_cell_8_kernel_regularizer_square_readvariableop_resource:	
identityЂ;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDlstm_8_lstm_cell_8_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02=
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOpе
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareSquareClstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2.
,lstm_8/lstm_cell_8/kernel/Regularizer/SquareЋ
+lstm_8/lstm_cell_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_8/lstm_cell_8/kernel/Regularizer/Constц
)lstm_8/lstm_cell_8/kernel/Regularizer/SumSum0lstm_8/lstm_cell_8/kernel/Regularizer/Square:y:04lstm_8/lstm_cell_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/Sum
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82-
+lstm_8/lstm_cell_8/kernel/Regularizer/mul/xш
)lstm_8/lstm_cell_8/kernel/Regularizer/mulMul4lstm_8/lstm_cell_8/kernel/Regularizer/mul/x:output:02lstm_8/lstm_cell_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_8/lstm_cell_8/kernel/Regularizer/mulw
IdentityIdentity-lstm_8/lstm_cell_8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp<^lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp;lstm_8/lstm_cell_8/kernel/Regularizer/Square/ReadVariableOp
Ј

Я
lstm_8_while_cond_162127*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_162127___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_162127___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_162127___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_162127___redundant_placeholder3
lstm_8_while_identity

lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
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
Є
Д
'__inference_lstm_8_layer_call_fn_163800

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCallџ
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1618352
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
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
Ж
'__inference_lstm_8_layer_call_fn_163767
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1605892
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
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
?
input_54
serving_default_input_5:0џџџџџџџџџA
	reshape_64
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ю
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
regularization_losses
	variables
		keras_api


signatures
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"
_tf_keras_sequential
У
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_rnn_layer
Л

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
Л

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
Ѕ
trainable_variables
regularization_losses
	variables
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
 "
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
Ъ

)layers
*layer_metrics
trainable_variables
+metrics
,layer_regularization_losses
regularization_losses
-non_trainable_variables
	variables
a__call__
b_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
с
.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
Й

3states

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables
	variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
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
­

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
regularization_losses
=non_trainable_variables
	variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_13/kernel
:2dense_13/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

>layers
?layer_metrics
trainable_variables
@metrics
Alayer_regularization_losses
regularization_losses
Bnon_trainable_variables
	variables
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
Dlayer_metrics
trainable_variables
Emetrics
Flayer_regularization_losses
regularization_losses
Gnon_trainable_variables
	variables
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
,:*	2lstm_8/lstm_cell_8/kernel
6:4	 2#lstm_8/lstm_cell_8/recurrent_kernel
&:$2lstm_8/lstm_cell_8/bias
<
0
1
2
3"
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
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
­

Ilayers
Jlayer_metrics
/trainable_variables
Kmetrics
Llayer_regularization_losses
0regularization_losses
Mnon_trainable_variables
1	variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
N
	Ntotal
	Ocount
P	variables
Q	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
n0"
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
&:$  2Adam/dense_12/kernel/m
 : 2Adam/dense_12/bias/m
&:$ 2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
1:/	2 Adam/lstm_8/lstm_cell_8/kernel/m
;:9	 2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
+:)2Adam/lstm_8/lstm_cell_8/bias/m
&:$  2Adam/dense_12/kernel/v
 : 2Adam/dense_12/bias/v
&:$ 2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
1:/	2 Adam/lstm_8/lstm_cell_8/kernel/v
;:9	 2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
+:)2Adam/lstm_8/lstm_cell_8/bias/v
ю2ы
H__inference_sequential_4_layer_call_and_return_conditional_losses_162283
H__inference_sequential_4_layer_call_and_return_conditional_losses_162612
H__inference_sequential_4_layer_call_and_return_conditional_losses_161957
H__inference_sequential_4_layer_call_and_return_conditional_losses_161985Р
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
2џ
-__inference_sequential_4_layer_call_fn_161489
-__inference_sequential_4_layer_call_fn_162631
-__inference_sequential_4_layer_call_fn_162650
-__inference_sequential_4_layer_call_fn_161929Р
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
ЬBЩ
!__inference__wrapped_model_160376input_5"
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
ы2ш
B__inference_lstm_8_layer_call_and_return_conditional_losses_162899
B__inference_lstm_8_layer_call_and_return_conditional_losses_163206
B__inference_lstm_8_layer_call_and_return_conditional_losses_163449
B__inference_lstm_8_layer_call_and_return_conditional_losses_163756е
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
џ2ќ
'__inference_lstm_8_layer_call_fn_163767
'__inference_lstm_8_layer_call_fn_163778
'__inference_lstm_8_layer_call_fn_163789
'__inference_lstm_8_layer_call_fn_163800е
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
ю2ы
D__inference_dense_12_layer_call_and_return_conditional_losses_163811Ђ
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
г2а
)__inference_dense_12_layer_call_fn_163820Ђ
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
ю2ы
D__inference_dense_13_layer_call_and_return_conditional_losses_163830Ђ
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
г2а
)__inference_dense_13_layer_call_fn_163839Ђ
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
E__inference_reshape_6_layer_call_and_return_conditional_losses_163852Ђ
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
*__inference_reshape_6_layer_call_fn_163857Ђ
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
ЫBШ
$__inference_signature_wrapper_162018input_5"
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
ж2г
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_163944
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_164057О
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
 2
,__inference_lstm_cell_8_layer_call_fn_164074
,__inference_lstm_cell_8_layer_call_fn_164091О
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
Г2А
__inference_loss_fn_0_164102
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
annotationsЊ *Ђ 
!__inference__wrapped_model_160376z&('4Ђ1
*Ђ'
%"
input_5џџџџџџџџџ
Њ "9Њ6
4
	reshape_6'$
	reshape_6џџџџџџџџџЄ
D__inference_dense_12_layer_call_and_return_conditional_losses_163811\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
)__inference_dense_12_layer_call_fn_163820O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
D__inference_dense_13_layer_call_and_return_conditional_losses_163830\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_13_layer_call_fn_163839O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ;
__inference_loss_fn_0_164102&Ђ

Ђ 
Њ " У
B__inference_lstm_8_layer_call_and_return_conditional_losses_162899}&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 У
B__inference_lstm_8_layer_call_and_return_conditional_losses_163206}&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 Г
B__inference_lstm_8_layer_call_and_return_conditional_losses_163449m&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 Г
B__inference_lstm_8_layer_call_and_return_conditional_losses_163756m&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 
'__inference_lstm_8_layer_call_fn_163767p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
'__inference_lstm_8_layer_call_fn_163778p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ 
'__inference_lstm_8_layer_call_fn_163789`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
'__inference_lstm_8_layer_call_fn_163800`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ Щ
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_163944§&('Ђ}
vЂs
 
inputsџџџџџџџџџ
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
 Щ
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_164057§&('Ђ}
vЂs
 
inputsџџџџџџџџџ
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
 
,__inference_lstm_cell_8_layer_call_fn_164074э&('Ђ}
vЂs
 
inputsџџџџџџџџџ
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
1/1џџџџџџџџџ 
,__inference_lstm_cell_8_layer_call_fn_164091э&('Ђ}
vЂs
 
inputsџџџџџџџџџ
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
1/1џџџџџџџџџ Ѕ
E__inference_reshape_6_layer_call_and_return_conditional_losses_163852\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 }
*__inference_reshape_6_layer_call_fn_163857O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџО
H__inference_sequential_4_layer_call_and_return_conditional_losses_161957r&('<Ђ9
2Ђ/
%"
input_5џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 О
H__inference_sequential_4_layer_call_and_return_conditional_losses_161985r&('<Ђ9
2Ђ/
%"
input_5џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
H__inference_sequential_4_layer_call_and_return_conditional_losses_162283q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
H__inference_sequential_4_layer_call_and_return_conditional_losses_162612q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 
-__inference_sequential_4_layer_call_fn_161489e&('<Ђ9
2Ђ/
%"
input_5џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_161929e&('<Ђ9
2Ђ/
%"
input_5џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_162631d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_162650d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЎ
$__inference_signature_wrapper_162018&('?Ђ<
Ђ 
5Њ2
0
input_5%"
input_5џџџџџџџџџ"9Њ6
4
	reshape_6'$
	reshape_6џџџџџџџџџ