іІ$
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8УН#
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:  *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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
З
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_namelstm/lstm_cell/kernel
А
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	А*
dtype0
Ы
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*0
shared_name!lstm/lstm_cell/recurrent_kernel
Ф
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	 А*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
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
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:  *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
Х
Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*-
shared_nameAdam/lstm/lstm_cell/kernel/m
О
0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes
:	А*
dtype0
©
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
Ґ
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Н
Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/m
Ж
.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes	
:А*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:  *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
Х
Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*-
shared_nameAdam/lstm/lstm_cell/kernel/v
О
0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes
:	А*
dtype0
©
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
Ґ
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Н
Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/v
Ж
.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Г+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Њ*
valueі*B±* B™*
у
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
	variables
trainable_variables
)layer_metrics
*non_trainable_variables
+metrics

,layers
regularization_losses
-layer_regularization_losses
 
О
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
є
	variables
trainable_variables
3layer_metrics
4metrics
5non_trainable_variables

6layers

7states
regularization_losses
8layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
	variables
trainable_variables
9metrics
:layer_metrics
;non_trainable_variables

<layers
regularization_losses
=layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
	variables
trainable_variables
>metrics
?layer_metrics
@non_trainable_variables

Alayers
regularization_losses
Blayer_regularization_losses
 
 
 
≠
	variables
trainable_variables
Cmetrics
Dlayer_metrics
Enon_trainable_variables

Flayers
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
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 

H0

0
1
2
3
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
/	variables
0trainable_variables
Imetrics
Jlayer_metrics
Knon_trainable_variables

Llayers
1regularization_losses
Mlayer_regularization_losses
 
 
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
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
В
serving_default_input_1Placeholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
»
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm/lstm_cell/kernellstm/lstm_cell/biaslstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*
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
GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_44669
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
™
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*)
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
GPU 2J 8В *'
f"R 
__inference__traced_save_46860
х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*(
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_46954ƒУ"
ќ∆
Ц
E__inference_sequential_layer_call_and_return_conditional_losses_44972

inputs?
,lstm_lstm_cell_split_readvariableop_resource:	А=
.lstm_lstm_cell_split_1_readvariableop_resource:	А9
&lstm_lstm_cell_readvariableop_resource:	 А6
$dense_matmul_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstЙ

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstС
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permЙ
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1В
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackЖ
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1Ж
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2М
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1П
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm/TensorArrayV2/element_shape∆
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2…
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorВ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackЖ
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1Ж
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Ъ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm/strided_slice_2Г
lstm/lstm_cell/ones_like/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell/ones_like/ShapeЕ
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
lstm/lstm_cell/ones_like/Constј
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/ones_likeВ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimЄ
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#lstm/lstm_cell/split/ReadVariableOpг
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm/lstm_cell/split®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMulђ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_1ђ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_2ђ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_3Ж
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lstm/lstm_cell/split_1/split_dimЇ
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%lstm/lstm_cell/split_1/ReadVariableOpџ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm/lstm_cell/split_1ѓ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAddµ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_1µ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_2µ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_3Щ
lstm/lstm_cell/mulMullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mulЭ
lstm/lstm_cell/mul_1Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_1Э
lstm/lstm_cell/mul_2Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_2Э
lstm/lstm_cell/mul_3Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_3¶
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm/lstm_cell/ReadVariableOpЩ
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm/lstm_cell/strided_slice/stackЭ
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell/strided_slice/stack_1Э
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice/stack_2÷
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm/lstm_cell/strided_slice≠
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_4І
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/addЕ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid™
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_1Э
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell/strided_slice_1/stack°
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&lstm/lstm_cell/strided_slice_1/stack_1°
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_2в
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_1±
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_1:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_5≠
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_1Л
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid_1Ъ
lstm/lstm_cell/mul_4Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_4™
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_2Э
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm/lstm_cell/strided_slice_2/stack°
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2(
&lstm/lstm_cell/strided_slice_2/stack_1°
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_2/stack_2в
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_2±
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_2:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_6≠
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_2~
lstm/lstm_cell/ReluRelulstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Relu§
lstm/lstm_cell/mul_5Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_5Ы
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_4:z:0lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_3™
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_3Э
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm/lstm_cell/strided_slice_3/stack°
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm/lstm_cell/strided_slice_3/stack_1°
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_3/stack_2в
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_3±
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_3:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_7≠
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_4Л
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid_2В
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Relu_1®
lstm/lstm_cell/mul_6Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_6Щ
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2$
"lstm/TensorArrayV2_1/element_shapeћ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/timeЙ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЅ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_44817*!
condR
lstm_while_cond_44816*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2

lstm/whileњ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeь
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackЛ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm/strided_slice_3/stackЖ
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1Ж
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2Є
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm/strided_slice_3Г
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permє
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense/MatMul/ReadVariableOpЬ
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

dense/Relu•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2»
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЭ
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape/Reshapeа
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mulw
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityз
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
їu
б
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46742

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
 *UU’?2
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
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2пѕС2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_1/ShapeЎ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2іУ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_2/ShapeЎ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2оџ_2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_3/Shapeў
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2З«њ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6—
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muld
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

Identity_2В
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
Ѓ
т
)__inference_lstm_cell_layer_call_fn_46531

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCallњ
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_431512
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
ћQ
я
D__inference_lstm_cell_layer_call_and_return_conditional_losses_43151

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6—
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muld
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

Identity_2В
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
№Q
б
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46629

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6—
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muld
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

Identity_2В
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
эz
€
while_body_45461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ь
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul†
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1†
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2†
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
Ю
±
$__inference_lstm_layer_call_fn_45340

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallь
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_440602
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–
Њ
while_cond_45460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_45460___redundant_placeholder03
/while_while_cond_45460___redundant_placeholder13
/while_while_cond_45460___redundant_placeholder23
/while_while_cond_45460___redundant_placeholder3
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
© 
–
E__inference_sequential_layer_call_and_return_conditional_losses_44123

inputs

lstm_44061:	А

lstm_44063:	А

lstm_44065:	 А
dense_44080:  
dense_44082: 
dense_1_44096: 
dense_1_44098:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЛ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs
lstm_44061
lstm_44063
lstm_44065*
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_440602
lstm/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_44080dense_44082*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_440792
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44096dense_1_44098*
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
GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_440952!
dense_1/StatefulPartitionedCallц
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_441142
reshape/PartitionedCallЊ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
lstm_44061*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityй
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≈Ы
В
?__inference_lstm_layer_call_and_return_conditional_losses_44060

inputs:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likex
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Е
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulЙ
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1Й
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2Й
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_43927*
condR
while_cond_43926*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’@
У
__inference__traced_save_46860
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
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
ShardedFilenameц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*И
valueюBыB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¬
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesО
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
…: :  : : :: : : : : :	А:	 А:А: : :  : : ::	А:	 А:А:  : : ::	А:	 А:А: 2(
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
:	А:%!

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
:	А:%!

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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:

_output_shapes
: 
тЗ
Ѓ

lstm_while_body_44817&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	АE
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АA
.lstm_while_lstm_cell_readvariableop_resource_0:	 А
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	АC
4lstm_while_lstm_cell_split_1_readvariableop_resource:	А?
,lstm_while_lstm_cell_readvariableop_resource:	 АИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpЌ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeс
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemФ
$lstm/while/lstm_cell/ones_like/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell/ones_like/ShapeС
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm/while/lstm_cell/ones_like/ConstЎ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/ones_likeО
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimћ
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02+
)lstm/while/lstm_cell/split/ReadVariableOpы
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm/while/lstm_cell/split“
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul÷
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_1÷
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_2÷
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_3Т
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm/while/lstm_cell/split_1/split_dimќ
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02-
+lstm/while/lstm_cell/split_1/ReadVariableOpу
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm/while/lstm_cell/split_1«
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/BiasAddЌ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_1Ќ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_2Ќ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_3∞
lstm/while/lstm_cell/mulMullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mulі
lstm/while/lstm_cell/mul_1Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_1і
lstm/while/lstm_cell/mul_2Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_2і
lstm/while/lstm_cell/mul_3Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_3Ї
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#lstm/while/lstm_cell/ReadVariableOp•
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/while/lstm_cell/strided_slice/stack©
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell/strided_slice/stack_1©
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice/stack_2ъ
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm/while/lstm_cell/strided_slice≈
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_4њ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/addЧ
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/SigmoidЊ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_1©
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell/strided_slice_1/stack≠
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm/while/lstm_cell/strided_slice_1/stack_1≠
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_2Ж
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_1…
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_1:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_5≈
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_1Э
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/Sigmoid_1ѓ
lstm/while/lstm_cell/mul_4Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_4Њ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_2©
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm/while/lstm_cell/strided_slice_2/stack≠
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm/while/lstm_cell/strided_slice_2/stack_1≠
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_2/stack_2Ж
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_2…
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_2:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_6≈
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_2Р
lstm/while/lstm_cell/ReluRelulstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/ReluЉ
lstm/while/lstm_cell/mul_5Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_5≥
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_4:z:0lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_3Њ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_3©
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm/while/lstm_cell/strided_slice_3/stack≠
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm/while/lstm_cell/strided_slice_3/stack_1≠
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_3/stack_2Ж
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_3…
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_3:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_7≈
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_4Э
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/Sigmoid_2Ф
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/Relu_1ј
lstm/while/lstm_cell/mul_6Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_6ц
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/yП
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЧ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Б
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2Ѓ
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ю
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_6:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/Identity_4Ю
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/Identity_5№
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 
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
Е
^
B__inference_reshape_layer_call_and_return_conditional_losses_44114

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
≈Ы
В
?__inference_lstm_layer_call_and_return_conditional_losses_46144

inputs:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likex
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Е
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulЙ
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1Й
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2Й
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_46011*
condR
while_cond_46010*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ	
£
*__inference_sequential_layer_call_fn_44707

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЇ
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_445442
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ґ
≥
$__inference_lstm_layer_call_fn_45329
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallю
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_435372
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
–
Њ
while_cond_44320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_44320___redundant_placeholder03
/while_while_cond_44320___redundant_placeholder13
/while_while_cond_44320___redundant_placeholder23
/while_while_cond_44320___redundant_placeholder3
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
ь$
«
while_body_43165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_43189_0:	А&
while_lstm_cell_43191_0:	А*
while_lstm_cell_43193_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_43189:	А$
while_lstm_cell_43191:	А(
while_lstm_cell_43193:	 АИҐ'while/lstm_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЋ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_43189_0while_lstm_cell_43191_0while_lstm_cell_43193_0*
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_431512)
'while/lstm_cell/StatefulPartitionedCallф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
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
while/Identity_3°
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4°
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Д

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_43189while_lstm_cell_43189_0"0
while_lstm_cell_43191while_lstm_cell_43191_0"0
while_lstm_cell_43193while_lstm_cell_43193_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 
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
–
Њ
while_cond_46010
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_46010___redundant_placeholder03
/while_while_cond_46010___redundant_placeholder13
/while_while_cond_46010___redundant_placeholder23
/while_while_cond_46010___redundant_placeholder3
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
© 
–
E__inference_sequential_layer_call_and_return_conditional_losses_44544

inputs

lstm_44519:	А

lstm_44521:	А

lstm_44523:	 А
dense_44526:  
dense_44528: 
dense_1_44531: 
dense_1_44533:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpЛ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs
lstm_44519
lstm_44521
lstm_44523*
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_444862
lstm/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_44526dense_44528*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_440792
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44531dense_1_44533*
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
GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_440952!
dense_1/StatefulPartitionedCallц
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_441142
reshape/PartitionedCallЊ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
lstm_44519*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityй
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–
Њ
while_cond_43926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43926___redundant_placeholder03
/while_while_cond_43926___redundant_placeholder13
/while_while_cond_43926___redundant_placeholder23
/while_while_cond_43926___redundant_placeholder3
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
ђ 
—
E__inference_sequential_layer_call_and_return_conditional_losses_44608
input_1

lstm_44583:	А

lstm_44585:	А

lstm_44587:	 А
dense_44590:  
dense_44592: 
dense_1_44595: 
dense_1_44597:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpМ
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1
lstm_44583
lstm_44585
lstm_44587*
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_440602
lstm/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_44590dense_44592*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_440792
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44595dense_1_44597*
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
GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_440952!
dense_1/StatefulPartitionedCallц
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_441142
reshape/PartitionedCallЊ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
lstm_44583*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityй
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ђЂ
€
while_body_44321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeГ
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
while/lstm_cell/dropout/Constњ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/MulР
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/ShapeГ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2џГ∞26
4while/lstm_cell/dropout/random_uniform/RandomUniformХ
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2(
&while/lstm_cell/dropout/GreaterEqual/yю
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$while/lstm_cell/dropout/GreaterEqualѓ
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/CastЇ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/Mul_1З
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_1/Const≈
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_1/MulФ
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/ShapeИ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2дЌ28
6while/lstm_cell/dropout_1/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_1/GreaterEqual/yЖ
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_1/Cast¬
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_1/Mul_1З
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_2/Const≈
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_2/MulФ
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/ShapeЙ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ъЂд28
6while/lstm_cell/dropout_2/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_2/GreaterEqual/yЖ
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_2/Cast¬
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_2/Mul_1З
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_3/Const≈
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_3/MulФ
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/ShapeЙ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2е Џ28
6while/lstm_cell/dropout_3/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_3/GreaterEqual/yЖ
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_3/Cast¬
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_3/Mul_1Д
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ы
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul°
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1°
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2°
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
з	
Ґ
lstm_while_cond_45113&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_45113___redundant_placeholder0=
9lstm_while_lstm_while_cond_45113___redundant_placeholder1=
9lstm_while_lstm_while_cond_45113___redundant_placeholder2=
9lstm_while_lstm_while_cond_45113___redundant_placeholder3
lstm_while_identity
Й
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
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
Ш«
В
?__inference_lstm_layer_call_and_return_conditional_losses_44486

inputs:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout/ConstІ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeс
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2їіЏ20
.lstm_cell/dropout/random_uniform/RandomUniformЙ
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2"
 lstm_cell/dropout/GreaterEqual/yж
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_cell/dropout/GreaterEqualЭ
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/CastҐ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_1/Const≠
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/MulВ
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeч
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2МЉ№22
0lstm_cell/dropout_1/random_uniform/RandomUniformН
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_1/GreaterEqual/yо
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Cast™
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_2/Const≠
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/MulВ
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeч
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2£°Ы22
0lstm_cell/dropout_2/random_uniform/RandomUniformН
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_2/GreaterEqual/yо
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Cast™
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_3/Const≠
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/MulВ
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeч
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2…У 22
0lstm_cell/dropout_3/random_uniform/RandomUniformН
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_3/GreaterEqual/yо
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Cast™
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Mul_1x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Д
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulК
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1К
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2К
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_44321*
condR
while_cond_44320*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠Ђ
€
while_body_45736
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeГ
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
while/lstm_cell/dropout/Constњ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/MulР
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/ShapeГ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2¬№Ы26
4while/lstm_cell/dropout/random_uniform/RandomUniformХ
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2(
&while/lstm_cell/dropout/GreaterEqual/yю
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$while/lstm_cell/dropout/GreaterEqualѓ
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/CastЇ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/Mul_1З
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_1/Const≈
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_1/MulФ
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/ShapeЙ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2№лу28
6while/lstm_cell/dropout_1/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_1/GreaterEqual/yЖ
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_1/Cast¬
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_1/Mul_1З
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_2/Const≈
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_2/MulФ
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/ShapeЙ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Рцн28
6while/lstm_cell/dropout_2/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_2/GreaterEqual/yЖ
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_2/Cast¬
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_2/Mul_1З
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_3/Const≈
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_3/MulФ
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/ShapeЙ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2≈ЫМ28
6while/lstm_cell/dropout_3/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_3/GreaterEqual/yЖ
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_3/Cast¬
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_3/Mul_1Д
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ы
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul°
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1°
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2°
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
ђ 
—
E__inference_sequential_layer_call_and_return_conditional_losses_44636
input_1

lstm_44611:	А

lstm_44613:	А

lstm_44615:	 А
dense_44618:  
dense_44620: 
dense_1_44623: 
dense_1_44625:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpМ
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1
lstm_44611
lstm_44613
lstm_44615*
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_444862
lstm/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_44618dense_44620*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_440792
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44623dense_1_44625*
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
GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_440952!
dense_1/StatefulPartitionedCallц
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_441142
reshape/PartitionedCallЊ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
lstm_44611*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityй
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ћ
љ
__inference_loss_fn_0_46753S
@lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource:	А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpф
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentity)lstm/lstm_cell/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityИ
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp
ЂЂ
€
while_body_46286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeГ
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
while/lstm_cell/dropout/Constњ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/MulР
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/ShapeВ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2¬Ц#26
4while/lstm_cell/dropout/random_uniform/RandomUniformХ
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2(
&while/lstm_cell/dropout/GreaterEqual/yю
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$while/lstm_cell/dropout/GreaterEqualѓ
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/CastЇ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout/Mul_1З
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_1/Const≈
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_1/MulФ
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/ShapeЙ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЙЮМ28
6while/lstm_cell/dropout_1/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_1/GreaterEqual/yЖ
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_1/Cast¬
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_1/Mul_1З
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_2/Const≈
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_2/MulФ
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/ShapeЙ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2НИ≠28
6while/lstm_cell/dropout_2/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_2/GreaterEqual/yЖ
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_2/Cast¬
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_2/Mul_1З
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell/dropout_3/Const≈
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/dropout_3/MulФ
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/ShapeИ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Ъџ28
6while/lstm_cell/dropout_3/random_uniform/RandomUniformЩ
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell/dropout_3/GreaterEqual/yЖ
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell/dropout_3/Cast¬
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell/dropout_3/Mul_1Д
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ы
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul°
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1°
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2°
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
ќг
ю
 __inference__wrapped_model_43027
input_1J
7sequential_lstm_lstm_cell_split_readvariableop_resource:	АH
9sequential_lstm_lstm_cell_split_1_readvariableop_resource:	АD
1sequential_lstm_lstm_cell_readvariableop_resource:	 АA
/sequential_dense_matmul_readvariableop_resource:  >
0sequential_dense_biasadd_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource: @
2sequential_dense_1_biasadd_readvariableop_resource:
identityИҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpҐ(sequential/lstm/lstm_cell/ReadVariableOpҐ*sequential/lstm/lstm_cell/ReadVariableOp_1Ґ*sequential/lstm/lstm_cell/ReadVariableOp_2Ґ*sequential/lstm/lstm_cell/ReadVariableOp_3Ґ.sequential/lstm/lstm_cell/split/ReadVariableOpҐ0sequential/lstm/lstm_cell/split_1/ReadVariableOpҐsequential/lstm/whilee
sequential/lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
sequential/lstm/ShapeФ
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stackШ
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1Ш
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2¬
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice|
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/zeros/mul/yђ
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
sequential/lstm/zeros/Less/yІ
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/LessВ
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/lstm/zeros/packed/1√
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/Constµ
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/lstm/zerosА
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/zeros_1/mul/y≤
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mulГ
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
sequential/lstm/zeros_1/Less/yѓ
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/LessЖ
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/lstm/zeros_1/packed/1…
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packedГ
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/Constљ
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/lstm/zeros_1Х
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
sequential/lstm/transpose/permЂ
sequential/lstm/transpose	Transposeinput_1'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
sequential/lstm/transpose
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:2
sequential/lstm/Shape_1Ш
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_1/stackЬ
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_1Ь
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_2ќ
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sequential/lstm/strided_slice_1•
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2-
+sequential/lstm/TensorArrayV2/element_shapeт
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/lstm/TensorArrayV2я
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2G
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeЄ
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorШ
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_2/stackЬ
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_1Ь
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_2№
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2!
sequential/lstm/strided_slice_2§
)sequential/lstm/lstm_cell/ones_like/ShapeShapesequential/lstm/zeros:output:0*
T0*
_output_shapes
:2+
)sequential/lstm/lstm_cell/ones_like/ShapeЫ
)sequential/lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)sequential/lstm/lstm_cell/ones_like/Constм
#sequential/lstm/lstm_cell/ones_likeFill2sequential/lstm/lstm_cell/ones_like/Shape:output:02sequential/lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/ones_likeШ
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/lstm/lstm_cell/split/split_dimў
.sequential/lstm/lstm_cell/split/ReadVariableOpReadVariableOp7sequential_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype020
.sequential/lstm/lstm_cell/split/ReadVariableOpП
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:06sequential/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2!
sequential/lstm/lstm_cell/split‘
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 sequential/lstm/lstm_cell/MatMulЎ
"sequential/lstm/lstm_cell/MatMul_1MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_1Ў
"sequential/lstm/lstm_cell/MatMul_2MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_2Ў
"sequential/lstm/lstm_cell/MatMul_3MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_3Ь
+sequential/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/lstm/lstm_cell/split_1/split_dimџ
0sequential/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype022
0sequential/lstm/lstm_cell/split_1/ReadVariableOpЗ
!sequential/lstm/lstm_cell/split_1Split4sequential/lstm/lstm_cell/split_1/split_dim:output:08sequential/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2#
!sequential/lstm/lstm_cell/split_1џ
!sequential/lstm/lstm_cell/BiasAddBiasAdd*sequential/lstm/lstm_cell/MatMul:product:0*sequential/lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!sequential/lstm/lstm_cell/BiasAddб
#sequential/lstm/lstm_cell/BiasAdd_1BiasAdd,sequential/lstm/lstm_cell/MatMul_1:product:0*sequential/lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/BiasAdd_1б
#sequential/lstm/lstm_cell/BiasAdd_2BiasAdd,sequential/lstm/lstm_cell/MatMul_2:product:0*sequential/lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/BiasAdd_2б
#sequential/lstm/lstm_cell/BiasAdd_3BiasAdd,sequential/lstm/lstm_cell/MatMul_3:product:0*sequential/lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/BiasAdd_3≈
sequential/lstm/lstm_cell/mulMulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/lstm/lstm_cell/mul…
sequential/lstm/lstm_cell/mul_1Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_1…
sequential/lstm/lstm_cell/mul_2Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_2…
sequential/lstm/lstm_cell/mul_3Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_3«
(sequential/lstm/lstm_cell/ReadVariableOpReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02*
(sequential/lstm/lstm_cell/ReadVariableOpѓ
-sequential/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-sequential/lstm/lstm_cell/strided_slice/stack≥
/sequential/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/lstm/lstm_cell/strided_slice/stack_1≥
/sequential/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/lstm/lstm_cell/strided_slice/stack_2Ш
'sequential/lstm/lstm_cell/strided_sliceStridedSlice0sequential/lstm/lstm_cell/ReadVariableOp:value:06sequential/lstm/lstm_cell/strided_slice/stack:output:08sequential/lstm/lstm_cell/strided_slice/stack_1:output:08sequential/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2)
'sequential/lstm/lstm_cell/strided_sliceў
"sequential/lstm/lstm_cell/MatMul_4MatMul!sequential/lstm/lstm_cell/mul:z:00sequential/lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_4”
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/BiasAdd:output:0,sequential/lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/lstm/lstm_cell/add¶
!sequential/lstm/lstm_cell/SigmoidSigmoid!sequential/lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!sequential/lstm/lstm_cell/SigmoidЋ
*sequential/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_1≥
/sequential/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/lstm/lstm_cell/strided_slice_1/stackЈ
1sequential/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   23
1sequential/lstm/lstm_cell/strided_slice_1/stack_1Ј
1sequential/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_1/stack_2§
)sequential/lstm/lstm_cell/strided_slice_1StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_1:value:08sequential/lstm/lstm_cell/strided_slice_1/stack:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_1Ё
"sequential/lstm/lstm_cell/MatMul_5MatMul#sequential/lstm/lstm_cell/mul_1:z:02sequential/lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_5ў
sequential/lstm/lstm_cell/add_1AddV2,sequential/lstm/lstm_cell/BiasAdd_1:output:0,sequential/lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/add_1ђ
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid#sequential/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/Sigmoid_1∆
sequential/lstm/lstm_cell/mul_4Mul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_4Ћ
*sequential/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_2≥
/sequential/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   21
/sequential/lstm/lstm_cell/strided_slice_2/stackЈ
1sequential/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   23
1sequential/lstm/lstm_cell/strided_slice_2/stack_1Ј
1sequential/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_2/stack_2§
)sequential/lstm/lstm_cell/strided_slice_2StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_2:value:08sequential/lstm/lstm_cell/strided_slice_2/stack:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_2Ё
"sequential/lstm/lstm_cell/MatMul_6MatMul#sequential/lstm/lstm_cell/mul_2:z:02sequential/lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_6ў
sequential/lstm/lstm_cell/add_2AddV2,sequential/lstm/lstm_cell/BiasAdd_2:output:0,sequential/lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/add_2Я
sequential/lstm/lstm_cell/ReluRelu#sequential/lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
sequential/lstm/lstm_cell/Relu–
sequential/lstm/lstm_cell/mul_5Mul%sequential/lstm/lstm_cell/Sigmoid:y:0,sequential/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_5«
sequential/lstm/lstm_cell/add_3AddV2#sequential/lstm/lstm_cell/mul_4:z:0#sequential/lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/add_3Ћ
*sequential/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_3≥
/sequential/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   21
/sequential/lstm/lstm_cell/strided_slice_3/stackЈ
1sequential/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1sequential/lstm/lstm_cell/strided_slice_3/stack_1Ј
1sequential/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_3/stack_2§
)sequential/lstm/lstm_cell/strided_slice_3StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_3:value:08sequential/lstm/lstm_cell/strided_slice_3/stack:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_3Ё
"sequential/lstm/lstm_cell/MatMul_7MatMul#sequential/lstm/lstm_cell/mul_3:z:02sequential/lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"sequential/lstm/lstm_cell/MatMul_7ў
sequential/lstm/lstm_cell/add_4AddV2,sequential/lstm/lstm_cell/BiasAdd_3:output:0,sequential/lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/add_4ђ
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid#sequential/lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/lstm_cell/Sigmoid_2£
 sequential/lstm/lstm_cell/Relu_1Relu#sequential/lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 sequential/lstm/lstm_cell/Relu_1‘
sequential/lstm/lstm_cell/mul_6Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0.sequential/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
sequential/lstm/lstm_cell/mul_6ѓ
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2/
-sequential/lstm/TensorArrayV2_1/element_shapeш
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
sequential/lstm/TensorArrayV2_1n
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/timeЯ
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2*
(sequential/lstm/while/maximum_iterationsК
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential/lstm/while/loop_counterж
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_lstm_lstm_cell_split_readvariableop_resource9sequential_lstm_lstm_cell_split_1_readvariableop_resource1sequential_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 sequential_lstm_while_body_42878*,
cond$R"
 sequential_lstm_while_cond_42877*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
sequential/lstm/while’
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2B
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape®
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype024
2sequential/lstm/TensorArrayV2Stack/TensorListStack°
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2'
%sequential/lstm/strided_slice_3/stackЬ
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential/lstm/strided_slice_3/stack_1Ь
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_3/stack_2ъ
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2!
sequential/lstm/strided_slice_3Щ
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/lstm/transpose_1/permе
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
sequential/lstm/transpose_1Ж
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/runtimeј
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02(
&sequential/dense/MatMul/ReadVariableOp»
sequential/dense/MatMulMatMul(sequential/lstm/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/dense/MatMulњ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp≈
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/dense/BiasAddЛ
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential/dense/Relu∆
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp…
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential/dense_1/BiasAddЗ
sequential/reshape/ShapeShape#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential/reshape/ShapeЪ
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stackЮ
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1Ю
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2‘
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_sliceК
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1К
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2€
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shape…
sequential/reshape/ReshapeReshape#sequential/dense_1/BiasAdd:output:0)sequential/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
sequential/reshape/ReshapeВ
IdentityIdentity#sequential/reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identity¶
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp)^sequential/lstm/lstm_cell/ReadVariableOp+^sequential/lstm/lstm_cell/ReadVariableOp_1+^sequential/lstm/lstm_cell/ReadVariableOp_2+^sequential/lstm/lstm_cell/ReadVariableOp_3/^sequential/lstm/lstm_cell/split/ReadVariableOp1^sequential/lstm/lstm_cell/split_1/ReadVariableOp^sequential/lstm/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2T
(sequential/lstm/lstm_cell/ReadVariableOp(sequential/lstm/lstm_cell/ReadVariableOp2X
*sequential/lstm/lstm_cell/ReadVariableOp_1*sequential/lstm/lstm_cell/ReadVariableOp_12X
*sequential/lstm/lstm_cell/ReadVariableOp_2*sequential/lstm/lstm_cell/ReadVariableOp_22X
*sequential/lstm/lstm_cell/ReadVariableOp_3*sequential/lstm/lstm_cell/ReadVariableOp_32`
.sequential/lstm/lstm_cell/split/ReadVariableOp.sequential/lstm/lstm_cell/split/ReadVariableOp2d
0sequential/lstm/lstm_cell/split_1/ReadVariableOp0sequential/lstm/lstm_cell/split_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
§

у
B__inference_dense_1_layer_call_and_return_conditional_losses_44095

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
≠u
я
D__inference_lstm_cell_layer_call_and_return_conditional_losses_43384

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
 *UU’?2
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
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Юяў2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_1/Shapeў
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Х£®2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_2/Shapeў
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2≤„≥2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
 *UU’?2
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
dropout_3/Shapeў
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Ф≈ц2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6—
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muld
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

Identity_2В
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
ыЫ
Д
?__inference_lstm_layer_call_and_return_conditional_losses_45594
inputs_0:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likex
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Е
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulЙ
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1Й
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2Й
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_45461*
condR
while_cond_45460*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
эz
€
while_body_46011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ь
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul†
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1†
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2†
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
§

у
B__inference_dense_1_layer_call_and_return_conditional_losses_46490

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
™z
±
!__inference__traced_restore_46954
file_prefix/
assignvariableop_dense_kernel:  +
assignvariableop_1_dense_bias: 3
!assignvariableop_2_dense_1_kernel: -
assignvariableop_3_dense_1_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ;
(assignvariableop_9_lstm_lstm_cell_kernel:	АF
3assignvariableop_10_lstm_lstm_cell_recurrent_kernel:	 А6
'assignvariableop_11_lstm_lstm_cell_bias:	А#
assignvariableop_12_total: #
assignvariableop_13_count: 9
'assignvariableop_14_adam_dense_kernel_m:  3
%assignvariableop_15_adam_dense_bias_m: ;
)assignvariableop_16_adam_dense_1_kernel_m: 5
'assignvariableop_17_adam_dense_1_bias_m:C
0assignvariableop_18_adam_lstm_lstm_cell_kernel_m:	АM
:assignvariableop_19_adam_lstm_lstm_cell_recurrent_kernel_m:	 А=
.assignvariableop_20_adam_lstm_lstm_cell_bias_m:	А9
'assignvariableop_21_adam_dense_kernel_v:  3
%assignvariableop_22_adam_dense_bias_v: ;
)assignvariableop_23_adam_dense_1_kernel_v: 5
'assignvariableop_24_adam_dense_1_bias_v:C
0assignvariableop_25_adam_lstm_lstm_cell_kernel_v:	АM
:assignvariableop_26_adam_lstm_lstm_cell_recurrent_kernel_v:	 А=
.assignvariableop_27_adam_lstm_lstm_cell_bias_v:	А
identity_29ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ь
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*И
valueюBыB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9≠
AssignVariableOp_9AssignVariableOp(assignvariableop_9_lstm_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ї
AssignVariableOp_10AssignVariableOp3assignvariableop_10_lstm_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ѓ
AssignVariableOp_11AssignVariableOp'assignvariableop_11_lstm_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*
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
Identity_14ѓ
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15≠
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ѓ
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Є
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_lstm_lstm_cell_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¬
AssignVariableOp_19AssignVariableOp:assignvariableop_19_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ґ
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_lstm_lstm_cell_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ѓ
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≠
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ѓ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Є
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_lstm_lstm_cell_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¬
AssignVariableOp_26AssignVariableOp:assignvariableop_26_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ґ
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_lstm_lstm_cell_bias_vIdentity_27:output:0"/device:CPU:0*
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
Ё	
§
*__inference_sequential_layer_call_fn_44140
input_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_441232
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ю

с
@__inference_dense_layer_call_and_return_conditional_losses_46471

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
ґ
≥
$__inference_lstm_layer_call_fn_45318
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallю
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_432402
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Џ	
£
*__inference_sequential_layer_call_fn_44688

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЇ
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_441232
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю

с
@__inference_dense_layer_call_and_return_conditional_losses_44079

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
≤
ю
 sequential_lstm_while_cond_42877<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1S
Osequential_lstm_while_sequential_lstm_while_cond_42877___redundant_placeholder0S
Osequential_lstm_while_sequential_lstm_while_cond_42877___redundant_placeholder1S
Osequential_lstm_while_sequential_lstm_while_cond_42877___redundant_placeholder2S
Osequential_lstm_while_sequential_lstm_while_cond_42877___redundant_placeholder3"
sequential_lstm_while_identity
ј
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: 2
sequential/lstm/while/LessН
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
sequential/lstm/while/Identity"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*(
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
–
Њ
while_cond_43461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43461___redundant_placeholder03
/while_while_cond_43461___redundant_placeholder13
/while_while_cond_43461___redundant_placeholder23
/while_while_cond_43461___redundant_placeholder3
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
Ш«
В
?__inference_lstm_layer_call_and_return_conditional_losses_46451

inputs:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout/ConstІ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeс
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ьию20
.lstm_cell/dropout/random_uniform/RandomUniformЙ
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2"
 lstm_cell/dropout/GreaterEqual/yж
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_cell/dropout/GreaterEqualЭ
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/CastҐ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_1/Const≠
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/MulВ
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeч
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2иЭУ22
0lstm_cell/dropout_1/random_uniform/RandomUniformН
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_1/GreaterEqual/yо
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Cast™
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_2/Const≠
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/MulВ
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeч
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2«ч÷22
0lstm_cell/dropout_2/random_uniform/RandomUniformН
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_2/GreaterEqual/yо
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Cast™
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_3/Const≠
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/MulВ
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeч
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2г≤П22
0lstm_cell/dropout_3/random_uniform/RandomUniformН
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_3/GreaterEqual/yо
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Cast™
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Mul_1x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Д
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulК
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1К
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2К
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_46286*
condR
while_cond_46285*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 P
ђ
?__inference_lstm_layer_call_and_return_conditional_losses_43240

inputs"
lstm_cell_43152:	А
lstm_cell_43154:	А"
lstm_cell_43156:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2З
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_43152lstm_cell_43154lstm_cell_43156*
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_431512#
!lstm_cell/StatefulPartitionedCallП
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
while/loop_counter≤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_43152lstm_cell_43154lstm_cell_43156*
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
bodyR
while_body_43165*
condR
while_cond_43164*K
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
runtime√
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_43152*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityі
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щц
Ц
E__inference_sequential_layer_call_and_return_conditional_losses_45301

inputs?
,lstm_lstm_cell_split_readvariableop_resource:	А=
.lstm_lstm_cell_split_1_readvariableop_resource:	А9
&lstm_lstm_cell_readvariableop_resource:	 А6
$dense_matmul_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstЙ

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstС
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permЙ
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1В
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackЖ
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1Ж
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2М
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1П
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 lstm/TensorArrayV2/element_shape∆
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2…
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorВ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackЖ
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1Ж
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Ъ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm/strided_slice_2Г
lstm/lstm_cell/ones_like/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell/ones_like/ShapeЕ
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
lstm/lstm_cell/ones_like/Constј
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/ones_likeБ
lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm/lstm_cell/dropout/Constї
lstm/lstm_cell/dropout/MulMul!lstm/lstm_cell/ones_like:output:0%lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout/MulН
lstm/lstm_cell/dropout/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm/lstm_cell/dropout/ShapeА
3lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform%lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2§ЪР25
3lstm/lstm_cell/dropout/random_uniform/RandomUniformУ
%lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2'
%lstm/lstm_cell/dropout/GreaterEqual/yъ
#lstm/lstm_cell/dropout/GreaterEqualGreaterEqual<lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:0.lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#lstm/lstm_cell/dropout/GreaterEqualђ
lstm/lstm_cell/dropout/CastCast'lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout/Castґ
lstm/lstm_cell/dropout/Mul_1Mullstm/lstm_cell/dropout/Mul:z:0lstm/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout/Mul_1Е
lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2 
lstm/lstm_cell/dropout_1/ConstЅ
lstm/lstm_cell/dropout_1/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_1/MulС
lstm/lstm_cell/dropout_1/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell/dropout_1/ShapeЕ
5lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Щ°>27
5lstm/lstm_cell/dropout_1/random_uniform/RandomUniformЧ
'lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2)
'lstm/lstm_cell/dropout_1/GreaterEqual/yВ
%lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm/lstm_cell/dropout_1/GreaterEqual≤
lstm/lstm_cell/dropout_1/CastCast)lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_1/CastЊ
lstm/lstm_cell/dropout_1/Mul_1Mul lstm/lstm_cell/dropout_1/Mul:z:0!lstm/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/lstm_cell/dropout_1/Mul_1Е
lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2 
lstm/lstm_cell/dropout_2/ConstЅ
lstm/lstm_cell/dropout_2/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_2/MulС
lstm/lstm_cell/dropout_2/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell/dropout_2/ShapeЖ
5lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ьЃљ27
5lstm/lstm_cell/dropout_2/random_uniform/RandomUniformЧ
'lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2)
'lstm/lstm_cell/dropout_2/GreaterEqual/yВ
%lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm/lstm_cell/dropout_2/GreaterEqual≤
lstm/lstm_cell/dropout_2/CastCast)lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_2/CastЊ
lstm/lstm_cell/dropout_2/Mul_1Mul lstm/lstm_cell/dropout_2/Mul:z:0!lstm/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/lstm_cell/dropout_2/Mul_1Е
lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2 
lstm/lstm_cell/dropout_3/ConstЅ
lstm/lstm_cell/dropout_3/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_3/MulС
lstm/lstm_cell/dropout_3/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell/dropout_3/ShapeЖ
5lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2тЁ№27
5lstm/lstm_cell/dropout_3/random_uniform/RandomUniformЧ
'lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2)
'lstm/lstm_cell/dropout_3/GreaterEqual/yВ
%lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%lstm/lstm_cell/dropout_3/GreaterEqual≤
lstm/lstm_cell/dropout_3/CastCast)lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/dropout_3/CastЊ
lstm/lstm_cell/dropout_3/Mul_1Mul lstm/lstm_cell/dropout_3/Mul:z:0!lstm/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/lstm_cell/dropout_3/Mul_1В
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimЄ
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#lstm/lstm_cell/split/ReadVariableOpг
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm/lstm_cell/split®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMulђ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_1ђ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_2ђ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_3Ж
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lstm/lstm_cell/split_1/split_dimЇ
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%lstm/lstm_cell/split_1/ReadVariableOpџ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm/lstm_cell/split_1ѓ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAddµ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_1µ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_2µ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/BiasAdd_3Ш
lstm/lstm_cell/mulMullstm/zeros:output:0 lstm/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mulЮ
lstm/lstm_cell/mul_1Mullstm/zeros:output:0"lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_1Ю
lstm/lstm_cell/mul_2Mullstm/zeros:output:0"lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_2Ю
lstm/lstm_cell/mul_3Mullstm/zeros:output:0"lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_3¶
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm/lstm_cell/ReadVariableOpЩ
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm/lstm_cell/strided_slice/stackЭ
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell/strided_slice/stack_1Э
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice/stack_2÷
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm/lstm_cell/strided_slice≠
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_4І
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/addЕ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid™
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_1Э
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell/strided_slice_1/stack°
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&lstm/lstm_cell/strided_slice_1/stack_1°
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_2в
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_1±
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_1:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_5≠
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_1Л
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid_1Ъ
lstm/lstm_cell/mul_4Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_4™
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_2Э
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm/lstm_cell/strided_slice_2/stack°
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2(
&lstm/lstm_cell/strided_slice_2/stack_1°
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_2/stack_2в
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_2±
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_2:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_6≠
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_2~
lstm/lstm_cell/ReluRelulstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Relu§
lstm/lstm_cell/mul_5Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_5Ы
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_4:z:0lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_3™
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02!
lstm/lstm_cell/ReadVariableOp_3Э
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm/lstm_cell/strided_slice_3/stack°
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm/lstm_cell/strided_slice_3/stack_1°
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_3/stack_2в
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_3±
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_3:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/MatMul_7≠
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/add_4Л
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Sigmoid_2В
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/Relu_1®
lstm/lstm_cell/mul_6Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/lstm_cell/mul_6Щ
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2$
"lstm/TensorArrayV2_1/element_shapeћ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/timeЙ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЅ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_45114*!
condR
lstm_while_cond_45113*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2

lstm/whileњ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeь
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackЛ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm/strided_slice_3/stackЖ
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1Ж
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2Є
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm/strided_slice_3Г
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permє
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense/MatMul/ReadVariableOpЬ
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

dense/Relu•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2»
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЭ
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape/Reshapeа
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/mulw
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityз
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_38^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
эz
€
while_body_43927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	А@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	А>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	 АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЕ
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/ShapeЗ
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2!
while/lstm_cell/ones_like/Constƒ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/ones_likeД
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimљ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02&
$while/lstm_cell/split/ReadVariableOpз
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell/splitЊ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_3И
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimњ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpя
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell/split_1≥
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAddє
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_1є
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_2є
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/BiasAdd_3Ь
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul†
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_1†
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_2†
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_3Ђ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02 
while/lstm_cell/ReadVariableOpЫ
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stackЯ
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice/stack_1Я
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2№
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_4Ђ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/addИ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoidѓ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_1Я
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2и
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_1О
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_1Ы
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_4ѓ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_2Я
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2и
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_2Б
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu®
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_5Я
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_3ѓ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell/ReadVariableOp_3Я
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2и
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/add_4О
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Sigmoid_2Е
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/Relu_1ђ
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell/mul_6Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
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
while/Identity_3К
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4К
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5і

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 
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
ХЉ
Ѓ

lstm_while_body_45114&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	АE
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АA
.lstm_while_lstm_cell_readvariableop_resource_0:	 А
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	АC
4lstm_while_lstm_cell_split_1_readvariableop_resource:	А?
,lstm_while_lstm_cell_readvariableop_resource:	 АИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpЌ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeс
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemФ
$lstm/while/lstm_cell/ones_like/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell/ones_like/ShapeС
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$lstm/while/lstm_cell/ones_like/ConstЎ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/ones_likeН
"lstm/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2$
"lstm/while/lstm_cell/dropout/Const”
 lstm/while/lstm_cell/dropout/MulMul'lstm/while/lstm_cell/ones_like:output:0+lstm/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm/while/lstm_cell/dropout/MulЯ
"lstm/while/lstm_cell/dropout/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm/while/lstm_cell/dropout/ShapeТ
9lstm/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform+lstm/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2∞Рх2;
9lstm/while/lstm_cell/dropout/random_uniform/RandomUniformЯ
+lstm/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2-
+lstm/while/lstm_cell/dropout/GreaterEqual/yТ
)lstm/while/lstm_cell/dropout/GreaterEqualGreaterEqualBlstm/while/lstm_cell/dropout/random_uniform/RandomUniform:output:04lstm/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)lstm/while/lstm_cell/dropout/GreaterEqualЊ
!lstm/while/lstm_cell/dropout/CastCast-lstm/while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!lstm/while/lstm_cell/dropout/Castќ
"lstm/while/lstm_cell/dropout/Mul_1Mul$lstm/while/lstm_cell/dropout/Mul:z:0%lstm/while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm/while/lstm_cell/dropout/Mul_1С
$lstm/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2&
$lstm/while/lstm_cell/dropout_1/Constў
"lstm/while/lstm_cell/dropout_1/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm/while/lstm_cell/dropout_1/Mul£
$lstm/while/lstm_cell/dropout_1/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell/dropout_1/ShapeШ
;lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЕШ≥2=
;lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform£
-lstm/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2/
-lstm/while/lstm_cell/dropout_1/GreaterEqual/yЪ
+lstm/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm/while/lstm_cell/dropout_1/GreaterEqualƒ
#lstm/while/lstm_cell/dropout_1/CastCast/lstm/while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm/while/lstm_cell/dropout_1/Cast÷
$lstm/while/lstm_cell/dropout_1/Mul_1Mul&lstm/while/lstm_cell/dropout_1/Mul:z:0'lstm/while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm/while/lstm_cell/dropout_1/Mul_1С
$lstm/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2&
$lstm/while/lstm_cell/dropout_2/Constў
"lstm/while/lstm_cell/dropout_2/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm/while/lstm_cell/dropout_2/Mul£
$lstm/while/lstm_cell/dropout_2/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell/dropout_2/ShapeШ
;lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2°Ъы2=
;lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform£
-lstm/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2/
-lstm/while/lstm_cell/dropout_2/GreaterEqual/yЪ
+lstm/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm/while/lstm_cell/dropout_2/GreaterEqualƒ
#lstm/while/lstm_cell/dropout_2/CastCast/lstm/while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm/while/lstm_cell/dropout_2/Cast÷
$lstm/while/lstm_cell/dropout_2/Mul_1Mul&lstm/while/lstm_cell/dropout_2/Mul:z:0'lstm/while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm/while/lstm_cell/dropout_2/Mul_1С
$lstm/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2&
$lstm/while/lstm_cell/dropout_3/Constў
"lstm/while/lstm_cell/dropout_3/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm/while/lstm_cell/dropout_3/Mul£
$lstm/while/lstm_cell/dropout_3/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell/dropout_3/ShapeЧ
;lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2СЬ[2=
;lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform£
-lstm/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2/
-lstm/while/lstm_cell/dropout_3/GreaterEqual/yЪ
+lstm/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+lstm/while/lstm_cell/dropout_3/GreaterEqualƒ
#lstm/while/lstm_cell/dropout_3/CastCast/lstm/while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2%
#lstm/while/lstm_cell/dropout_3/Cast÷
$lstm/while/lstm_cell/dropout_3/Mul_1Mul&lstm/while/lstm_cell/dropout_3/Mul:z:0'lstm/while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm/while/lstm_cell/dropout_3/Mul_1О
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimћ
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02+
)lstm/while/lstm_cell/split/ReadVariableOpы
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm/while/lstm_cell/split“
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul÷
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_1÷
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_2÷
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_3Т
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm/while/lstm_cell/split_1/split_dimќ
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02-
+lstm/while/lstm_cell/split_1/ReadVariableOpу
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm/while/lstm_cell/split_1«
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/BiasAddЌ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_1Ќ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_2Ќ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/BiasAdd_3ѓ
lstm/while/lstm_cell/mulMullstm_while_placeholder_2&lstm/while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mulµ
lstm/while/lstm_cell/mul_1Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_1µ
lstm/while/lstm_cell/mul_2Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_2µ
lstm/while/lstm_cell/mul_3Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_3Ї
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02%
#lstm/while/lstm_cell/ReadVariableOp•
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/while/lstm_cell/strided_slice/stack©
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell/strided_slice/stack_1©
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice/stack_2ъ
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm/while/lstm_cell/strided_slice≈
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_4њ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/addЧ
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/SigmoidЊ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_1©
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell/strided_slice_1/stack≠
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm/while/lstm_cell/strided_slice_1/stack_1≠
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_2Ж
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_1…
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_1:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_5≈
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_1Э
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/Sigmoid_1ѓ
lstm/while/lstm_cell/mul_4Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_4Њ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_2©
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm/while/lstm_cell/strided_slice_2/stack≠
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm/while/lstm_cell/strided_slice_2/stack_1≠
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_2/stack_2Ж
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_2…
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_2:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_6≈
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_2Р
lstm/while/lstm_cell/ReluRelulstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/ReluЉ
lstm/while/lstm_cell/mul_5Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_5≥
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_4:z:0lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_3Њ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_3©
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm/while/lstm_cell/strided_slice_3/stack≠
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm/while/lstm_cell/strided_slice_3/stack_1≠
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_3/stack_2Ж
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_3…
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_3:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/MatMul_7≈
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/add_4Э
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm/while/lstm_cell/Sigmoid_2Ф
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/Relu_1ј
lstm/while/lstm_cell/mul_6Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/lstm_cell/mul_6ц
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/yП
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЧ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Б
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2Ѓ
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ю
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_6:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/Identity_4Ю
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm/while/Identity_5№
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 
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
ј
C
'__inference_reshape_layer_call_fn_46495

inputs
identityƒ
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
GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_441142
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
Ѓ
т
)__inference_lstm_cell_layer_call_fn_46548

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCallњ
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_433842
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
Ќ«
Д
?__inference_lstm_layer_call_and_return_conditional_losses_45901
inputs_0:
'lstm_cell_split_readvariableop_resource:	А8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell/ones_like/Constђ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout/ConstІ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeс
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2сџ•20
.lstm_cell/dropout/random_uniform/RandomUniformЙ
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2"
 lstm_cell/dropout/GreaterEqual/yж
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_cell/dropout/GreaterEqualЭ
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/CastҐ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_1/Const≠
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/MulВ
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeч
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Д≥Э22
0lstm_cell/dropout_1/random_uniform/RandomUniformН
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_1/GreaterEqual/yо
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Cast™
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_2/Const≠
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/MulВ
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeц
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ВЃ{22
0lstm_cell/dropout_2/random_uniform/RandomUniformН
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_2/GreaterEqual/yо
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Cast™
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell/dropout_3/Const≠
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/MulВ
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeч
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Мкъ22
0lstm_cell/dropout_3/random_uniform/RandomUniformН
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell/dropout_3/GreaterEqual/yо
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Cast™
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/dropout_3/Mul_1x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim©
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype02 
lstm_cell/split/ReadVariableOpѕ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell/splitФ
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMulШ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_1Ш
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_2Ш
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЂ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 lstm_cell/split_1/ReadVariableOp«
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell/split_1Ы
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd°
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_1°
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_2°
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/BiasAdd_3Д
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mulК
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_1К
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_2К
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_3Ч
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOpП
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stackУ
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice/stack_1У
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Є
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_sliceЩ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_4У
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/SigmoidЫ
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_1У
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_1/stackЧ
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell/strided_slice_1/stack_1Ч
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2ƒ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_1Э
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_5Щ
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_1Ж
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_4Ы
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_2У
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_2/stackЧ
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell/strided_slice_2/stack_1Ч
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2ƒ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_2Э
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_6Щ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/ReluР
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_5З
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_3Ы
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell/ReadVariableOp_3У
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2!
lstm_cell/strided_slice_3/stackЧ
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1Ч
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2ƒ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell/strided_slice_3Э
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/MatMul_7Щ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/Relu_1Ф
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell/mul_6П
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
while/loop_counterц
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
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
bodyR
while_body_45736*
condR
while_cond_45735*K
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
runtimeџ
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity∆
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
±	
Э
#__inference_signature_wrapper_44669
input_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В *)
f$R"
 __inference__wrapped_model_430272
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
–
Њ
while_cond_45735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_45735___redundant_placeholder03
/while_while_cond_45735___redundant_placeholder13
/while_while_cond_45735___redundant_placeholder23
/while_while_cond_45735___redundant_placeholder3
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
Ё	
§
*__inference_sequential_layer_call_fn_44580
input_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_445442
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
й
Т
%__inference_dense_layer_call_fn_46460

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallр
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_440792
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
з	
Ґ
lstm_while_cond_44816&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_44816___redundant_placeholder0=
9lstm_while_lstm_while_cond_44816___redundant_placeholder1=
9lstm_while_lstm_while_cond_44816___redundant_placeholder2=
9lstm_while_lstm_while_cond_44816___redundant_placeholder3
lstm_while_identity
Й
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
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
 P
ђ
?__inference_lstm_layer_call_and_return_conditional_losses_43537

inputs"
lstm_cell_43449:	А
lstm_cell_43451:	А"
lstm_cell_43453:	 А
identityИҐ7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2З
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_43449lstm_cell_43451lstm_cell_43453*
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_433842#
!lstm_cell/StatefulPartitionedCallП
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
while/loop_counter≤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_43449lstm_cell_43451lstm_cell_43453*
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
bodyR
while_body_43462*
condR
while_cond_43461*K
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
runtime√
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_43449*
_output_shapes
:	А*
dtype029
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp…
(lstm/lstm_cell/kernel/Regularizer/SquareSquare?lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2*
(lstm/lstm_cell/kernel/Regularizer/Square£
'lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'lstm/lstm_cell/kernel/Regularizer/Const÷
%lstm/lstm_cell/kernel/Regularizer/SumSum,lstm/lstm_cell/kernel/Regularizer/Square:y:00lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/SumЧ
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82)
'lstm/lstm_cell/kernel/Regularizer/mul/xЎ
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:0.lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%lstm/lstm_cell/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityі
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2r
7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
•§
ѓ
 sequential_lstm_while_body_42878<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0:	АP
Asequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АL
9sequential_lstm_while_lstm_cell_readvariableop_resource_0:	 А"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorP
=sequential_lstm_while_lstm_cell_split_readvariableop_resource:	АN
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resource:	АJ
7sequential_lstm_while_lstm_cell_readvariableop_resource:	 АИҐ.sequential/lstm/while/lstm_cell/ReadVariableOpҐ0sequential/lstm/while/lstm_cell/ReadVariableOp_1Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_2Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_3Ґ4sequential/lstm/while/lstm_cell/split/ReadVariableOpҐ6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpг
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape≥
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02;
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemµ
/sequential/lstm/while/lstm_cell/ones_like/ShapeShape#sequential_lstm_while_placeholder_2*
T0*
_output_shapes
:21
/sequential/lstm/while/lstm_cell/ones_like/ShapeІ
/sequential/lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?21
/sequential/lstm/while/lstm_cell/ones_like/ConstД
)sequential/lstm/while/lstm_cell/ones_likeFill8sequential/lstm/while/lstm_cell/ones_like/Shape:output:08sequential/lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/ones_like§
/sequential/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential/lstm/while/lstm_cell/split/split_dimн
4sequential/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype026
4sequential/lstm/while/lstm_cell/split/ReadVariableOpІ
%sequential/lstm/while/lstm_cell/splitSplit8sequential/lstm/while/lstm_cell/split/split_dim:output:0<sequential/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2'
%sequential/lstm/while/lstm_cell/splitю
&sequential/lstm/while/lstm_cell/MatMulMatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential/lstm/while/lstm_cell/MatMulВ
(sequential/lstm/while/lstm_cell/MatMul_1MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_1В
(sequential/lstm/while/lstm_cell/MatMul_2MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_2В
(sequential/lstm/while/lstm_cell/MatMul_3MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_3®
1sequential/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential/lstm/while/lstm_cell/split_1/split_dimп
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype028
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpЯ
'sequential/lstm/while/lstm_cell/split_1Split:sequential/lstm/while/lstm_cell/split_1/split_dim:output:0>sequential/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2)
'sequential/lstm/while/lstm_cell/split_1у
'sequential/lstm/while/lstm_cell/BiasAddBiasAdd0sequential/lstm/while/lstm_cell/MatMul:product:00sequential/lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential/lstm/while/lstm_cell/BiasAddщ
)sequential/lstm/while/lstm_cell/BiasAdd_1BiasAdd2sequential/lstm/while/lstm_cell/MatMul_1:product:00sequential/lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/BiasAdd_1щ
)sequential/lstm/while/lstm_cell/BiasAdd_2BiasAdd2sequential/lstm/while/lstm_cell/MatMul_2:product:00sequential/lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/BiasAdd_2щ
)sequential/lstm/while/lstm_cell/BiasAdd_3BiasAdd2sequential/lstm/while/lstm_cell/MatMul_3:product:00sequential/lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/BiasAdd_3№
#sequential/lstm/while/lstm_cell/mulMul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/while/lstm_cell/mulа
%sequential/lstm/while/lstm_cell/mul_1Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_1а
%sequential/lstm/while/lstm_cell/mul_2Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_2а
%sequential/lstm/while/lstm_cell/mul_3Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_3џ
.sequential/lstm/while/lstm_cell/ReadVariableOpReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype020
.sequential/lstm/while/lstm_cell/ReadVariableOpї
3sequential/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/lstm/while/lstm_cell/strided_slice/stackњ
5sequential/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/lstm/while/lstm_cell/strided_slice/stack_1њ
5sequential/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/lstm/while/lstm_cell/strided_slice/stack_2Љ
-sequential/lstm/while/lstm_cell/strided_sliceStridedSlice6sequential/lstm/while/lstm_cell/ReadVariableOp:value:0<sequential/lstm/while/lstm_cell/strided_slice/stack:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_1:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2/
-sequential/lstm/while/lstm_cell/strided_sliceс
(sequential/lstm/while/lstm_cell/MatMul_4MatMul'sequential/lstm/while/lstm_cell/mul:z:06sequential/lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_4л
#sequential/lstm/while/lstm_cell/addAddV20sequential/lstm/while/lstm_cell/BiasAdd:output:02sequential/lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential/lstm/while/lstm_cell/addЄ
'sequential/lstm/while/lstm_cell/SigmoidSigmoid'sequential/lstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential/lstm/while/lstm_cell/Sigmoidя
0sequential/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_1њ
5sequential/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/lstm/while/lstm_cell/strided_slice_1/stack√
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   29
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1√
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2»
/sequential/lstm/while/lstm_cell/strided_slice_1StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_1:value:0>sequential/lstm/while/lstm_cell/strided_slice_1/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_1х
(sequential/lstm/while/lstm_cell/MatMul_5MatMul)sequential/lstm/while/lstm_cell/mul_1:z:08sequential/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_5с
%sequential/lstm/while/lstm_cell/add_1AddV22sequential/lstm/while/lstm_cell/BiasAdd_1:output:02sequential/lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/add_1Њ
)sequential/lstm/while/lstm_cell/Sigmoid_1Sigmoid)sequential/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/Sigmoid_1џ
%sequential/lstm/while/lstm_cell/mul_4Mul-sequential/lstm/while/lstm_cell/Sigmoid_1:y:0#sequential_lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_4я
0sequential/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_2њ
5sequential/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5sequential/lstm/while/lstm_cell/strided_slice_2/stack√
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   29
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1√
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2»
/sequential/lstm/while/lstm_cell/strided_slice_2StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_2:value:0>sequential/lstm/while/lstm_cell/strided_slice_2/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_2х
(sequential/lstm/while/lstm_cell/MatMul_6MatMul)sequential/lstm/while/lstm_cell/mul_2:z:08sequential/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_6с
%sequential/lstm/while/lstm_cell/add_2AddV22sequential/lstm/while/lstm_cell/BiasAdd_2:output:02sequential/lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/add_2±
$sequential/lstm/while/lstm_cell/ReluRelu)sequential/lstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential/lstm/while/lstm_cell/Reluи
%sequential/lstm/while/lstm_cell/mul_5Mul+sequential/lstm/while/lstm_cell/Sigmoid:y:02sequential/lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_5я
%sequential/lstm/while/lstm_cell/add_3AddV2)sequential/lstm/while/lstm_cell/mul_4:z:0)sequential/lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/add_3я
0sequential/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_3њ
5sequential/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   27
5sequential/lstm/while/lstm_cell/strided_slice_3/stack√
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1√
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2»
/sequential/lstm/while/lstm_cell/strided_slice_3StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_3:value:0>sequential/lstm/while/lstm_cell/strided_slice_3/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_3х
(sequential/lstm/while/lstm_cell/MatMul_7MatMul)sequential/lstm/while/lstm_cell/mul_3:z:08sequential/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential/lstm/while/lstm_cell/MatMul_7с
%sequential/lstm/while/lstm_cell/add_4AddV22sequential/lstm/while/lstm_cell/BiasAdd_3:output:02sequential/lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/add_4Њ
)sequential/lstm/while/lstm_cell/Sigmoid_2Sigmoid)sequential/lstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential/lstm/while/lstm_cell/Sigmoid_2µ
&sequential/lstm/while/lstm_cell/Relu_1Relu)sequential/lstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential/lstm/while/lstm_cell/Relu_1м
%sequential/lstm/while/lstm_cell/mul_6Mul-sequential/lstm/while/lstm_cell/Sigmoid_2:y:04sequential/lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential/lstm/while/lstm_cell/mul_6≠
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder)sequential/lstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02<
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItem|
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add/y©
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/addА
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add_1/y∆
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add_1Ђ
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: 2 
sequential/lstm/while/Identityќ
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations^sequential/lstm/while/NoOp*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_1≠
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_2Џ
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_3 
 sequential/lstm/while/Identity_4Identity)sequential/lstm/while/lstm_cell/mul_6:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 sequential/lstm/while/Identity_4 
 sequential/lstm/while/Identity_5Identity)sequential/lstm/while/lstm_cell/add_3:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 sequential/lstm/while/Identity_5і
sequential/lstm/while/NoOpNoOp/^sequential/lstm/while/lstm_cell/ReadVariableOp1^sequential/lstm/while/lstm_cell/ReadVariableOp_11^sequential/lstm/while/lstm_cell/ReadVariableOp_21^sequential/lstm/while/lstm_cell/ReadVariableOp_35^sequential/lstm/while/lstm_cell/split/ReadVariableOp7^sequential/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
sequential/lstm/while/NoOp"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"t
7sequential_lstm_while_lstm_cell_readvariableop_resource9sequential_lstm_while_lstm_cell_readvariableop_resource_0"Д
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resourceAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0"А
=sequential_lstm_while_lstm_cell_split_readvariableop_resource?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"и
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2`
.sequential/lstm/while/lstm_cell/ReadVariableOp.sequential/lstm/while/lstm_cell/ReadVariableOp2d
0sequential/lstm/while/lstm_cell/ReadVariableOp_10sequential/lstm/while/lstm_cell/ReadVariableOp_12d
0sequential/lstm/while/lstm_cell/ReadVariableOp_20sequential/lstm/while/lstm_cell/ReadVariableOp_22d
0sequential/lstm/while/lstm_cell/ReadVariableOp_30sequential/lstm/while/lstm_cell/ReadVariableOp_32l
4sequential/lstm/while/lstm_cell/split/ReadVariableOp4sequential/lstm/while/lstm_cell/split/ReadVariableOp2p
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp: 
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
–
Њ
while_cond_43164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43164___redundant_placeholder03
/while_while_cond_43164___redundant_placeholder13
/while_while_cond_43164___redundant_placeholder23
/while_while_cond_43164___redundant_placeholder3
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
н
Ф
'__inference_dense_1_layer_call_fn_46480

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallт
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
GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_440952
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
Е
^
B__inference_reshape_layer_call_and_return_conditional_losses_46508

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
Ю
±
$__inference_lstm_layer_call_fn_45351

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallь
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
GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_444862
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–
Њ
while_cond_46285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_46285___redundant_placeholder03
/while_while_cond_46285___redundant_placeholder13
/while_while_cond_46285___redundant_placeholder23
/while_while_cond_46285___redundant_placeholder3
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
ь$
«
while_body_43462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_43486_0:	А&
while_lstm_cell_43488_0:	А*
while_lstm_cell_43490_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_43486:	А$
while_lstm_cell_43488:	А(
while_lstm_cell_43490:	 АИҐ'while/lstm_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЋ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_43486_0while_lstm_cell_43488_0while_lstm_cell_43490_0*
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
GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_433842)
'while/lstm_cell/StatefulPartitionedCallф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
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
while/Identity_3°
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4°
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Д

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_43486while_lstm_cell_43486_0"0
while_lstm_cell_43488while_lstm_cell_43488_0"0
while_lstm_cell_43490while_lstm_cell_43490_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 
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
: "®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≤
serving_defaultЮ
?
input_14
serving_default_input_1:0€€€€€€€€€?
reshape4
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:и}
и
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
√
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
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
•
	variables
trainable_variables
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
 "
trackable_list_wrapper
 
	variables
trainable_variables
)layer_metrics
*non_trainable_variables
+metrics

,layers
regularization_losses
-layer_regularization_losses
`__call__
a_default_save_signature
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
б
.
state_size

&kernel
'recurrent_kernel
(bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
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
n0"
trackable_list_wrapper
є
	variables
trainable_variables
3layer_metrics
4metrics
5non_trainable_variables

6layers

7states
regularization_losses
8layer_regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:  2dense/kernel
: 2
dense/bias
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
	variables
trainable_variables
9metrics
:layer_metrics
;non_trainable_variables

<layers
regularization_losses
=layer_regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
	variables
trainable_variables
>metrics
?layer_metrics
@non_trainable_variables

Alayers
regularization_losses
Blayer_regularization_losses
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
	variables
trainable_variables
Cmetrics
Dlayer_metrics
Enon_trainable_variables

Flayers
regularization_losses
Glayer_regularization_losses
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
(:&	А2lstm/lstm_cell/kernel
2:0	 А2lstm/lstm_cell/recurrent_kernel
": А2lstm/lstm_cell/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
<
0
1
2
3"
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
n0"
trackable_list_wrapper
≠
/	variables
0trainable_variables
Imetrics
Jlayer_metrics
Knon_trainable_variables

Llayers
1regularization_losses
Mlayer_regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
'
n0"
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
#:!  2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
-:+	А2Adam/lstm/lstm_cell/kernel/m
7:5	 А2&Adam/lstm/lstm_cell/recurrent_kernel/m
':%А2Adam/lstm/lstm_cell/bias/m
#:!  2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
-:+	А2Adam/lstm/lstm_cell/kernel/v
7:5	 А2&Adam/lstm/lstm_cell/recurrent_kernel/v
':%А2Adam/lstm/lstm_cell/bias/v
ц2у
*__inference_sequential_layer_call_fn_44140
*__inference_sequential_layer_call_fn_44688
*__inference_sequential_layer_call_fn_44707
*__inference_sequential_layer_call_fn_44580ј
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
ЋB»
 __inference__wrapped_model_43027input_1"Ш
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
в2я
E__inference_sequential_layer_call_and_return_conditional_losses_44972
E__inference_sequential_layer_call_and_return_conditional_losses_45301
E__inference_sequential_layer_call_and_return_conditional_losses_44608
E__inference_sequential_layer_call_and_return_conditional_losses_44636ј
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
у2р
$__inference_lstm_layer_call_fn_45318
$__inference_lstm_layer_call_fn_45329
$__inference_lstm_layer_call_fn_45340
$__inference_lstm_layer_call_fn_45351’
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
я2№
?__inference_lstm_layer_call_and_return_conditional_losses_45594
?__inference_lstm_layer_call_and_return_conditional_losses_45901
?__inference_lstm_layer_call_and_return_conditional_losses_46144
?__inference_lstm_layer_call_and_return_conditional_losses_46451’
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
ѕ2ћ
%__inference_dense_layer_call_fn_46460Ґ
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
к2з
@__inference_dense_layer_call_and_return_conditional_losses_46471Ґ
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
—2ќ
'__inference_dense_1_layer_call_fn_46480Ґ
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
м2й
B__inference_dense_1_layer_call_and_return_conditional_losses_46490Ґ
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
—2ќ
'__inference_reshape_layer_call_fn_46495Ґ
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
м2й
B__inference_reshape_layer_call_and_return_conditional_losses_46508Ґ
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
 B«
#__inference_signature_wrapper_44669input_1"Ф
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
Ъ2Ч
)__inference_lstm_cell_layer_call_fn_46531
)__inference_lstm_cell_layer_call_fn_46548Њ
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
–2Ќ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46629
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46742Њ
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
≤2ѓ
__inference_loss_fn_0_46753П
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
annotations™ *Ґ Ъ
 __inference__wrapped_model_43027v&('4Ґ1
*Ґ'
%К"
input_1€€€€€€€€€
™ "5™2
0
reshape%К"
reshape€€€€€€€€€Ґ
B__inference_dense_1_layer_call_and_return_conditional_losses_46490\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_dense_1_layer_call_fn_46480O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€†
@__inference_dense_layer_call_and_return_conditional_losses_46471\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ x
%__inference_dense_layer_call_fn_46460O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ :
__inference_loss_fn_0_46753&Ґ

Ґ 
™ "К ∆
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46629э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ ∆
D__inference_lstm_cell_layer_call_and_return_conditional_losses_46742э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ Ы
)__inference_lstm_cell_layer_call_fn_46531н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
1/1€€€€€€€€€ Ы
)__inference_lstm_cell_layer_call_fn_46548н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
1/1€€€€€€€€€ ј
?__inference_lstm_layer_call_and_return_conditional_losses_45594}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ј
?__inference_lstm_layer_call_and_return_conditional_losses_45901}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ∞
?__inference_lstm_layer_call_and_return_conditional_losses_46144m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ∞
?__inference_lstm_layer_call_and_return_conditional_losses_46451m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ Ш
$__inference_lstm_layer_call_fn_45318p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ш
$__inference_lstm_layer_call_fn_45329p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ И
$__inference_lstm_layer_call_fn_45340`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ И
$__inference_lstm_layer_call_fn_45351`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ Ґ
B__inference_reshape_layer_call_and_return_conditional_losses_46508\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ z
'__inference_reshape_layer_call_fn_46495O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ї
E__inference_sequential_layer_call_and_return_conditional_losses_44608r&('<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ї
E__inference_sequential_layer_call_and_return_conditional_losses_44636r&('<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ї
E__inference_sequential_layer_call_and_return_conditional_losses_44972q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ї
E__inference_sequential_layer_call_and_return_conditional_losses_45301q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ У
*__inference_sequential_layer_call_fn_44140e&('<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€
p 

 
™ "К€€€€€€€€€У
*__inference_sequential_layer_call_fn_44580e&('<Ґ9
2Ґ/
%К"
input_1€€€€€€€€€
p

 
™ "К€€€€€€€€€Т
*__inference_sequential_layer_call_fn_44688d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Т
*__inference_sequential_layer_call_fn_44707d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€©
#__inference_signature_wrapper_44669Б&('?Ґ<
Ґ 
5™2
0
input_1%К"
input_1€€€€€€€€€"5™2
0
reshape%К"
reshape€€€€€€€€€