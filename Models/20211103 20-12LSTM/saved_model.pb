ВЕ'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8и&
|
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:  *
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
: *
dtype0
|
dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_123/kernel
u
$dense_123/kernel/Read/ReadVariableOpReadVariableOpdense_123/kernel*
_output_shapes

: *
dtype0
t
dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_123/bias
m
"dense_123/bias/Read/ReadVariableOpReadVariableOpdense_123/bias*
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

lstm_101/lstm_cell_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_namelstm_101/lstm_cell_101/kernel

1lstm_101/lstm_cell_101/kernel/Read/ReadVariableOpReadVariableOplstm_101/lstm_cell_101/kernel*
_output_shapes
:	*
dtype0
Ћ
'lstm_101/lstm_cell_101/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *8
shared_name)'lstm_101/lstm_cell_101/recurrent_kernel
Є
;lstm_101/lstm_cell_101/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_101/lstm_cell_101/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_101/lstm_cell_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelstm_101/lstm_cell_101/bias

/lstm_101/lstm_cell_101/bias/Read/ReadVariableOpReadVariableOplstm_101/lstm_cell_101/bias*
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

Adam/dense_122/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_122/kernel/m

+Adam/dense_122/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_122/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_122/bias/m
{
)Adam/dense_122/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/m*
_output_shapes
: *
dtype0

Adam/dense_123/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_123/kernel/m

+Adam/dense_123/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_123/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_123/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_123/bias/m
{
)Adam/dense_123/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_123/bias/m*
_output_shapes
:*
dtype0
Ѕ
$Adam/lstm_101/lstm_cell_101/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/lstm_101/lstm_cell_101/kernel/m

8Adam/lstm_101/lstm_cell_101/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/lstm_101/lstm_cell_101/kernel/m*
_output_shapes
:	*
dtype0
Й
.Adam/lstm_101/lstm_cell_101/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *?
shared_name0.Adam/lstm_101/lstm_cell_101/recurrent_kernel/m
В
BAdam/lstm_101/lstm_cell_101/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/lstm_101/lstm_cell_101/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

"Adam/lstm_101/lstm_cell_101/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/lstm_101/lstm_cell_101/bias/m

6Adam/lstm_101/lstm_cell_101/bias/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_101/lstm_cell_101/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_122/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_122/kernel/v

+Adam/dense_122/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_122/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_122/bias/v
{
)Adam/dense_122/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/v*
_output_shapes
: *
dtype0

Adam/dense_123/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_123/kernel/v

+Adam/dense_123/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_123/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_123/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_123/bias/v
{
)Adam/dense_123/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_123/bias/v*
_output_shapes
:*
dtype0
Ѕ
$Adam/lstm_101/lstm_cell_101/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/lstm_101/lstm_cell_101/kernel/v

8Adam/lstm_101/lstm_cell_101/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/lstm_101/lstm_cell_101/kernel/v*
_output_shapes
:	*
dtype0
Й
.Adam/lstm_101/lstm_cell_101/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *?
shared_name0.Adam/lstm_101/lstm_cell_101/recurrent_kernel/v
В
BAdam/lstm_101/lstm_cell_101/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/lstm_101/lstm_cell_101/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

"Adam/lstm_101/lstm_cell_101/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/lstm_101/lstm_cell_101/bias/v

6Adam/lstm_101/lstm_cell_101/bias/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_101/lstm_cell_101/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
е,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
value,B, Bќ+
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
\Z
VARIABLE_VALUEdense_122/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_122/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_123/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_123/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ca
VARIABLE_VALUElstm_101/lstm_cell_101/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'lstm_101/lstm_cell_101/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_101/lstm_cell_101/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

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
}
VARIABLE_VALUEAdam/dense_122/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_122/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_123/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_123/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/lstm_101/lstm_cell_101/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/lstm_101/lstm_cell_101/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_101/lstm_cell_101/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_122/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_122/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_123/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_123/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/lstm_101/lstm_cell_101/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/lstm_101/lstm_cell_101/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_101/lstm_cell_101/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_42Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
я
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_42lstm_101/lstm_cell_101/kernellstm_101/lstm_cell_101/bias'lstm_101/lstm_cell_101/recurrent_kerneldense_122/kerneldense_122/biasdense_123/kerneldense_123/bias*
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
%__inference_signature_wrapper_3239363
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_122/kernel/Read/ReadVariableOp"dense_122/bias/Read/ReadVariableOp$dense_123/kernel/Read/ReadVariableOp"dense_123/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1lstm_101/lstm_cell_101/kernel/Read/ReadVariableOp;lstm_101/lstm_cell_101/recurrent_kernel/Read/ReadVariableOp/lstm_101/lstm_cell_101/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_122/kernel/m/Read/ReadVariableOp)Adam/dense_122/bias/m/Read/ReadVariableOp+Adam/dense_123/kernel/m/Read/ReadVariableOp)Adam/dense_123/bias/m/Read/ReadVariableOp8Adam/lstm_101/lstm_cell_101/kernel/m/Read/ReadVariableOpBAdam/lstm_101/lstm_cell_101/recurrent_kernel/m/Read/ReadVariableOp6Adam/lstm_101/lstm_cell_101/bias/m/Read/ReadVariableOp+Adam/dense_122/kernel/v/Read/ReadVariableOp)Adam/dense_122/bias/v/Read/ReadVariableOp+Adam/dense_123/kernel/v/Read/ReadVariableOp)Adam/dense_123/bias/v/Read/ReadVariableOp8Adam/lstm_101/lstm_cell_101/kernel/v/Read/ReadVariableOpBAdam/lstm_101/lstm_cell_101/recurrent_kernel/v/Read/ReadVariableOp6Adam/lstm_101/lstm_cell_101/bias/v/Read/ReadVariableOpConst*)
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
 __inference__traced_save_3241589
у
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_122/kerneldense_122/biasdense_123/kerneldense_123/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_101/lstm_cell_101/kernel'lstm_101/lstm_cell_101/recurrent_kernellstm_101/lstm_cell_101/biastotalcountAdam/dense_122/kernel/mAdam/dense_122/bias/mAdam/dense_123/kernel/mAdam/dense_123/bias/m$Adam/lstm_101/lstm_cell_101/kernel/m.Adam/lstm_101/lstm_cell_101/recurrent_kernel/m"Adam/lstm_101/lstm_cell_101/bias/mAdam/dense_122/kernel/vAdam/dense_122/bias/vAdam/dense_123/kernel/vAdam/dense_123/bias/v$Adam/lstm_101/lstm_cell_101/kernel/v.Adam/lstm_101/lstm_cell_101/recurrent_kernel/v"Adam/lstm_101/lstm_cell_101/bias/v*(
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
#__inference__traced_restore_3241683љ%
ѕ

+__inference_dense_122_layer_call_fn_3241166

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallі
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
GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_32387372
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
щ

ќ
lstm_101_while_cond_3239813.
*lstm_101_while_lstm_101_while_loop_counter4
0lstm_101_while_lstm_101_while_maximum_iterations
lstm_101_while_placeholder 
lstm_101_while_placeholder_1 
lstm_101_while_placeholder_2 
lstm_101_while_placeholder_30
,lstm_101_while_less_lstm_101_strided_slice_1G
Clstm_101_while_lstm_101_while_cond_3239813___redundant_placeholder0G
Clstm_101_while_lstm_101_while_cond_3239813___redundant_placeholder1G
Clstm_101_while_lstm_101_while_cond_3239813___redundant_placeholder2G
Clstm_101_while_lstm_101_while_cond_3239813___redundant_placeholder3
lstm_101_while_identity

lstm_101/while/LessLesslstm_101_while_placeholder,lstm_101_while_less_lstm_101_strided_slice_1*
T0*
_output_shapes
: 2
lstm_101/while/Lessx
lstm_101/while/IdentityIdentitylstm_101/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_101/while/Identity";
lstm_101_while_identity lstm_101/while/Identity:output:0*(
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
к,
Р
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239220

inputs#
lstm_101_3239189:	
lstm_101_3239191:	#
lstm_101_3239193:	 #
dense_122_3239196:  
dense_122_3239198: #
dense_123_3239201: 
dense_123_3239203:
identityЂ!dense_122/StatefulPartitionedCallЂ!dense_123/StatefulPartitionedCallЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ lstm_101/StatefulPartitionedCallЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЋ
 lstm_101/StatefulPartitionedCallStatefulPartitionedCallinputslstm_101_3239189lstm_101_3239191lstm_101_3239193*
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32391562"
 lstm_101/StatefulPartitionedCallП
!dense_122/StatefulPartitionedCallStatefulPartitionedCall)lstm_101/StatefulPartitionedCall:output:0dense_122_3239196dense_122_3239198*
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
GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_32387372#
!dense_122/StatefulPartitionedCallР
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_3239201dense_123_3239203*
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
GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_32387592#
!dense_123/StatefulPartitionedCall
reshape_61/PartitionedCallPartitionedCall*dense_123/StatefulPartitionedCall:output:0*
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_32387782
reshape_61/PartitionedCallд
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_101_3239189*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulВ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_3239203*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mul
IdentityIdentity#reshape_61/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall1^dense_123/bias/Regularizer/Square/ReadVariableOp!^lstm_101/StatefulPartitionedCall@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2D
 lstm_101/StatefulPartitionedCall lstm_101/StatefulPartitionedCall2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 Є
Ж
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240300
inputs_0>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileF
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3240167*
condR
while_cond_3240166*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Т
Й
*__inference_lstm_101_layer_call_fn_3240024
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32378982
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

ї
F__inference_dense_122_layer_call_and_return_conditional_losses_3238737

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
гv
э
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3238042

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
seed2Ывд2&
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
seed2вЙЊ2(
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
seed22(
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
seed2нЌЅ2(
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
mul_6с
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
ReadVariableOp_3ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2,
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
&
ё
while_body_3238120
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_101_3238144_0:	,
while_lstm_cell_101_3238146_0:	0
while_lstm_cell_101_3238148_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_101_3238144:	*
while_lstm_cell_101_3238146:	.
while_lstm_cell_101_3238148:	 Ђ+while/lstm_cell_101/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_101/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_101_3238144_0while_lstm_cell_101_3238146_0while_lstm_cell_101_3238148_0*
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32380422-
+while/lstm_cell_101/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_101/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_101/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_101/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_101/StatefulPartitionedCall*"
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
while_lstm_cell_101_3238144while_lstm_cell_101_3238144_0"<
while_lstm_cell_101_3238146while_lstm_cell_101_3238146_0"<
while_lstm_cell_101_3238148while_lstm_cell_101_3238148_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2Z
+while/lstm_cell_101/StatefulPartitionedCall+while/lstm_cell_101/StatefulPartitionedCall: 
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

ї
F__inference_dense_122_layer_call_and_return_conditional_losses_3241177

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
тv
я
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241471

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
seed2ю2&
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
seed2ёр'2(
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
seed2j2(
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
seed2ЙщД2(
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
mul_6с
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
ReadVariableOp_3ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2,
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
ъЃ
Д
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240850

inputs>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileD
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3240717*
condR
while_cond_3240716*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ТЕ
Б	
while_body_3240442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
!while/lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2#
!while/lstm_cell_101/dropout/ConstЯ
while/lstm_cell_101/dropout/MulMul&while/lstm_cell_101/ones_like:output:0*while/lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_101/dropout/Mul
!while/lstm_cell_101/dropout/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_101/dropout/Shape
8while/lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЋЭЈ2:
8while/lstm_cell_101/dropout/random_uniform/RandomUniform
*while/lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2,
*while/lstm_cell_101/dropout/GreaterEqual/y
(while/lstm_cell_101/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_101/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_101/dropout/GreaterEqualЛ
 while/lstm_cell_101/dropout/CastCast,while/lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_101/dropout/CastЪ
!while/lstm_cell_101/dropout/Mul_1Mul#while/lstm_cell_101/dropout/Mul:z:0$while/lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout/Mul_1
#while/lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_1/Constе
!while/lstm_cell_101/dropout_1/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_1/Mul 
#while/lstm_cell_101/dropout_1/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_1/Shape
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2№32<
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_1/GreaterEqual/y
*while/lstm_cell_101/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_1/GreaterEqualС
"while/lstm_cell_101/dropout_1/CastCast.while/lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_1/Castв
#while/lstm_cell_101/dropout_1/Mul_1Mul%while/lstm_cell_101/dropout_1/Mul:z:0&while/lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_1/Mul_1
#while/lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_2/Constе
!while/lstm_cell_101/dropout_2/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_2/Mul 
#while/lstm_cell_101/dropout_2/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_2/Shape
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЪФ2<
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_2/GreaterEqual/y
*while/lstm_cell_101/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_2/GreaterEqualС
"while/lstm_cell_101/dropout_2/CastCast.while/lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_2/Castв
#while/lstm_cell_101/dropout_2/Mul_1Mul%while/lstm_cell_101/dropout_2/Mul:z:0&while/lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_2/Mul_1
#while/lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_3/Constе
!while/lstm_cell_101/dropout_3/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_3/Mul 
#while/lstm_cell_101/dropout_3/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_3/Shape
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2єт­2<
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_3/GreaterEqual/y
*while/lstm_cell_101/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_3/GreaterEqualС
"while/lstm_cell_101/dropout_3/CastCast.while/lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_3/Castв
#while/lstm_cell_101/dropout_3/Mul_1Mul%while/lstm_cell_101/dropout_3/Mul:z:0&while/lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_3/Mul_1
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ї
while/lstm_cell_101/mulMulwhile_placeholder_2%while/lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul­
while/lstm_cell_101/mul_1Mulwhile_placeholder_2'while/lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1­
while/lstm_cell_101/mul_2Mulwhile_placeholder_2'while/lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2­
while/lstm_cell_101/mul_3Mulwhile_placeholder_2'while/lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
S
я
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241358

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
mul_6с
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
ReadVariableOp_3ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2,
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
Њ
З
*__inference_lstm_101_layer_call_fn_3240046

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32387182
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
К
ј
/__inference_lstm_cell_101_layer_call_fn_3241260

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32378092
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
зR
а
E__inference_lstm_101_layer_call_and_return_conditional_losses_3237898

inputs(
lstm_cell_101_3237810:	$
lstm_cell_101_3237812:	(
lstm_cell_101_3237814:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂ%lstm_cell_101/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ї
%lstm_cell_101/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_101_3237810lstm_cell_101_3237812lstm_cell_101_3237814*
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32378092'
%lstm_cell_101/StatefulPartitionedCall
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
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_101_3237810lstm_cell_101_3237812lstm_cell_101_3237814*
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
while_body_3237823*
condR
while_cond_3237822*K
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
runtimeй
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_101_3237810*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityР
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp&^lstm_cell_101/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2N
%lstm_cell_101/StatefulPartitionedCall%lstm_cell_101/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ	
Њ
/__inference_sequential_41_layer_call_fn_3239256
input_42
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_42unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
J__inference_sequential_41_layer_call_and_return_conditional_losses_32392202
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
input_42
ћR
э
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3237809

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
mul_6с
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muld
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

Identity_2
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
ReadVariableOp_3ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2,
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
while_cond_3238584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3238584___redundant_placeholder05
1while_while_cond_3238584___redundant_placeholder15
1while_while_cond_3238584___redundant_placeholder25
1while_while_cond_3238584___redundant_placeholder3
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
while_cond_3238990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3238990___redundant_placeholder05
1while_while_cond_3238990___redundant_placeholder15
1while_while_cond_3238990___redundant_placeholder25
1while_while_cond_3238990___redundant_placeholder3
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
while_cond_3240441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3240441___redundant_placeholder05
1while_while_cond_3240441___redundant_placeholder15
1while_while_cond_3240441___redundant_placeholder25
1while_while_cond_3240441___redundant_placeholder3
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
К
ј
/__inference_lstm_cell_101_layer_call_fn_3241277

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32380422
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
о|

#__inference__traced_restore_3241683
file_prefix3
!assignvariableop_dense_122_kernel:  /
!assignvariableop_1_dense_122_bias: 5
#assignvariableop_2_dense_123_kernel: /
!assignvariableop_3_dense_123_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: C
0assignvariableop_9_lstm_101_lstm_cell_101_kernel:	N
;assignvariableop_10_lstm_101_lstm_cell_101_recurrent_kernel:	 >
/assignvariableop_11_lstm_101_lstm_cell_101_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: =
+assignvariableop_14_adam_dense_122_kernel_m:  7
)assignvariableop_15_adam_dense_122_bias_m: =
+assignvariableop_16_adam_dense_123_kernel_m: 7
)assignvariableop_17_adam_dense_123_bias_m:K
8assignvariableop_18_adam_lstm_101_lstm_cell_101_kernel_m:	U
Bassignvariableop_19_adam_lstm_101_lstm_cell_101_recurrent_kernel_m:	 E
6assignvariableop_20_adam_lstm_101_lstm_cell_101_bias_m:	=
+assignvariableop_21_adam_dense_122_kernel_v:  7
)assignvariableop_22_adam_dense_122_bias_v: =
+assignvariableop_23_adam_dense_123_kernel_v: 7
)assignvariableop_24_adam_dense_123_bias_v:K
8assignvariableop_25_adam_lstm_101_lstm_cell_101_kernel_v:	U
Bassignvariableop_26_adam_lstm_101_lstm_cell_101_recurrent_kernel_v:	 E
6assignvariableop_27_adam_lstm_101_lstm_cell_101_bias_v:	
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_122_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_122_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_123_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_123_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9Е
AssignVariableOp_9AssignVariableOp0assignvariableop_9_lstm_101_lstm_cell_101_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10У
AssignVariableOp_10AssignVariableOp;assignvariableop_10_lstm_101_lstm_cell_101_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11З
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_101_lstm_cell_101_biasIdentity_11:output:0"/device:CPU:0*
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
Identity_14Г
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_dense_122_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Б
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_122_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Г
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_123_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Б
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_123_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Р
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_lstm_101_lstm_cell_101_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ъ
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_lstm_101_lstm_cell_101_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20О
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_101_lstm_cell_101_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Г
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_122_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Б
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_122_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_123_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_123_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Р
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_lstm_101_lstm_cell_101_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ъ
AssignVariableOp_26AssignVariableOpBassignvariableop_26_adam_lstm_101_lstm_cell_101_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27О
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_101_lstm_cell_101_bias_vIdentity_27:output:0"/device:CPU:0*
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
Щв
Д
E__inference_lstm_101_layer_call_and_return_conditional_losses_3241157

inputs>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileD
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout/ConstЗ
lstm_cell_101/dropout/MulMul lstm_cell_101/ones_like:output:0$lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul
lstm_cell_101/dropout/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout/Shapeћ
2lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЉЃ24
2lstm_cell_101/dropout/random_uniform/RandomUniform
$lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_101/dropout/GreaterEqual/yі
"lstm_cell_101/dropout/GreaterEqualGreaterEqual;lstm_cell_101/dropout/random_uniform/RandomUniform:output:0-lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_101/dropout/GreaterEqualЉ
lstm_cell_101/dropout/CastCast&lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/CastВ
lstm_cell_101/dropout/Mul_1Mullstm_cell_101/dropout/Mul:z:0lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul_1
lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_1/ConstН
lstm_cell_101/dropout_1/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul
lstm_cell_101/dropout_1/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_1/Shape
4lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ф­26
4lstm_cell_101/dropout_1/random_uniform/RandomUniform
&lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_1/GreaterEqual/yў
$lstm_cell_101/dropout_1/GreaterEqualGreaterEqual=lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_1/GreaterEqualЏ
lstm_cell_101/dropout_1/CastCast(lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/CastК
lstm_cell_101/dropout_1/Mul_1Mullstm_cell_101/dropout_1/Mul:z:0 lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul_1
lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_2/ConstН
lstm_cell_101/dropout_2/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul
lstm_cell_101/dropout_2/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_2/Shape
4lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЛМ26
4lstm_cell_101/dropout_2/random_uniform/RandomUniform
&lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_2/GreaterEqual/yў
$lstm_cell_101/dropout_2/GreaterEqualGreaterEqual=lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_2/GreaterEqualЏ
lstm_cell_101/dropout_2/CastCast(lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/CastК
lstm_cell_101/dropout_2/Mul_1Mullstm_cell_101/dropout_2/Mul:z:0 lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul_1
lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_3/ConstН
lstm_cell_101/dropout_3/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul
lstm_cell_101/dropout_3/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_3/Shape
4lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ПЯ26
4lstm_cell_101/dropout_3/random_uniform/RandomUniform
&lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_3/GreaterEqual/yў
$lstm_cell_101/dropout_3/GreaterEqualGreaterEqual=lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_3/GreaterEqualЏ
lstm_cell_101/dropout_3/CastCast(lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/CastК
lstm_cell_101/dropout_3/Mul_1Mullstm_cell_101/dropout_3/Mul:z:0 lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul_1
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0!lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0!lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0!lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3240992*
condR
while_cond_3240991*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Њ
F__inference_dense_123_layer_call_and_return_conditional_losses_3241208

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_123/bias/Regularizer/Square/ReadVariableOp
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
BiasAddР
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_123/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
И	
 
%__inference_signature_wrapper_3239363
input_42
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_42unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
"__inference__wrapped_model_32376852
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
input_42
ф	
Ј
/__inference_sequential_41_layer_call_fn_3239382

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
J__inference_sequential_41_layer_call_and_return_conditional_losses_32387932
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
Ъ
H
,__inference_reshape_61_layer_call_fn_3241213

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
G__inference_reshape_61_layer_call_and_return_conditional_losses_32387782
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
C

 __inference__traced_save_3241589
file_prefix/
+savev2_dense_122_kernel_read_readvariableop-
)savev2_dense_122_bias_read_readvariableop/
+savev2_dense_123_kernel_read_readvariableop-
)savev2_dense_123_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_lstm_101_lstm_cell_101_kernel_read_readvariableopF
Bsavev2_lstm_101_lstm_cell_101_recurrent_kernel_read_readvariableop:
6savev2_lstm_101_lstm_cell_101_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_122_kernel_m_read_readvariableop4
0savev2_adam_dense_122_bias_m_read_readvariableop6
2savev2_adam_dense_123_kernel_m_read_readvariableop4
0savev2_adam_dense_123_bias_m_read_readvariableopC
?savev2_adam_lstm_101_lstm_cell_101_kernel_m_read_readvariableopM
Isavev2_adam_lstm_101_lstm_cell_101_recurrent_kernel_m_read_readvariableopA
=savev2_adam_lstm_101_lstm_cell_101_bias_m_read_readvariableop6
2savev2_adam_dense_122_kernel_v_read_readvariableop4
0savev2_adam_dense_122_bias_v_read_readvariableop6
2savev2_adam_dense_123_kernel_v_read_readvariableop4
0savev2_adam_dense_123_bias_v_read_readvariableopC
?savev2_adam_lstm_101_lstm_cell_101_kernel_v_read_readvariableopM
Isavev2_adam_lstm_101_lstm_cell_101_recurrent_kernel_v_read_readvariableopA
=savev2_adam_lstm_101_lstm_cell_101_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesњ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_122_kernel_read_readvariableop)savev2_dense_122_bias_read_readvariableop+savev2_dense_123_kernel_read_readvariableop)savev2_dense_123_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_lstm_101_lstm_cell_101_kernel_read_readvariableopBsavev2_lstm_101_lstm_cell_101_recurrent_kernel_read_readvariableop6savev2_lstm_101_lstm_cell_101_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_122_kernel_m_read_readvariableop0savev2_adam_dense_122_bias_m_read_readvariableop2savev2_adam_dense_123_kernel_m_read_readvariableop0savev2_adam_dense_123_bias_m_read_readvariableop?savev2_adam_lstm_101_lstm_cell_101_kernel_m_read_readvariableopIsavev2_adam_lstm_101_lstm_cell_101_recurrent_kernel_m_read_readvariableop=savev2_adam_lstm_101_lstm_cell_101_bias_m_read_readvariableop2savev2_adam_dense_122_kernel_v_read_readvariableop0savev2_adam_dense_122_bias_v_read_readvariableop2savev2_adam_dense_123_kernel_v_read_readvariableop0savev2_adam_dense_123_bias_v_read_readvariableop?savev2_adam_lstm_101_lstm_cell_101_kernel_v_read_readvariableopIsavev2_adam_lstm_101_lstm_cell_101_recurrent_kernel_v_read_readvariableop=savev2_adam_lstm_101_lstm_cell_101_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ъ	
Њ
/__inference_sequential_41_layer_call_fn_3238810
input_42
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_42unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
J__inference_sequential_41_layer_call_and_return_conditional_losses_32387932
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
input_42
хг
ь
lstm_101_while_body_3239814.
*lstm_101_while_lstm_101_while_loop_counter4
0lstm_101_while_lstm_101_while_maximum_iterations
lstm_101_while_placeholder 
lstm_101_while_placeholder_1 
lstm_101_while_placeholder_2 
lstm_101_while_placeholder_3-
)lstm_101_while_lstm_101_strided_slice_1_0i
elstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0:	M
>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0:	I
6lstm_101_while_lstm_cell_101_readvariableop_resource_0:	 
lstm_101_while_identity
lstm_101_while_identity_1
lstm_101_while_identity_2
lstm_101_while_identity_3
lstm_101_while_identity_4
lstm_101_while_identity_5+
'lstm_101_while_lstm_101_strided_slice_1g
clstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensorM
:lstm_101_while_lstm_cell_101_split_readvariableop_resource:	K
<lstm_101_while_lstm_cell_101_split_1_readvariableop_resource:	G
4lstm_101_while_lstm_cell_101_readvariableop_resource:	 Ђ+lstm_101/while/lstm_cell_101/ReadVariableOpЂ-lstm_101/while/lstm_cell_101/ReadVariableOp_1Ђ-lstm_101/while/lstm_cell_101/ReadVariableOp_2Ђ-lstm_101/while/lstm_cell_101/ReadVariableOp_3Ђ1lstm_101/while/lstm_cell_101/split/ReadVariableOpЂ3lstm_101/while/lstm_cell_101/split_1/ReadVariableOpе
@lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2B
@lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shape
2lstm_101/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0lstm_101_while_placeholderIlstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype024
2lstm_101/while/TensorArrayV2Read/TensorListGetItemЈ
,lstm_101/while/lstm_cell_101/ones_like/ShapeShapelstm_101_while_placeholder_2*
T0*
_output_shapes
:2.
,lstm_101/while/lstm_cell_101/ones_like/ShapeЁ
,lstm_101/while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,lstm_101/while/lstm_cell_101/ones_like/Constј
&lstm_101/while/lstm_cell_101/ones_likeFill5lstm_101/while/lstm_cell_101/ones_like/Shape:output:05lstm_101/while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/ones_like
*lstm_101/while/lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_101/while/lstm_cell_101/dropout/Constѓ
(lstm_101/while/lstm_cell_101/dropout/MulMul/lstm_101/while/lstm_cell_101/ones_like:output:03lstm_101/while/lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_101/while/lstm_cell_101/dropout/MulЗ
*lstm_101/while/lstm_cell_101/dropout/ShapeShape/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_101/while/lstm_cell_101/dropout/ShapeЈ
Alstm_101/while/lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform3lstm_101/while/lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2вїЊ2C
Alstm_101/while/lstm_cell_101/dropout/random_uniform/RandomUniformЏ
3lstm_101/while/lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_101/while/lstm_cell_101/dropout/GreaterEqual/yВ
1lstm_101/while/lstm_cell_101/dropout/GreaterEqualGreaterEqualJlstm_101/while/lstm_cell_101/dropout/random_uniform/RandomUniform:output:0<lstm_101/while/lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_101/while/lstm_cell_101/dropout/GreaterEqualж
)lstm_101/while/lstm_cell_101/dropout/CastCast5lstm_101/while/lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_101/while/lstm_cell_101/dropout/Castю
*lstm_101/while/lstm_cell_101/dropout/Mul_1Mul,lstm_101/while/lstm_cell_101/dropout/Mul:z:0-lstm_101/while/lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_101/while/lstm_cell_101/dropout/Mul_1Ё
,lstm_101/while/lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2.
,lstm_101/while/lstm_cell_101/dropout_1/Constљ
*lstm_101/while/lstm_cell_101/dropout_1/MulMul/lstm_101/while/lstm_cell_101/ones_like:output:05lstm_101/while/lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_101/while/lstm_cell_101/dropout_1/MulЛ
,lstm_101/while/lstm_cell_101/dropout_1/ShapeShape/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_101/while/lstm_cell_101/dropout_1/ShapeЎ
Clstm_101/while/lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform5lstm_101/while/lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЇэХ2E
Clstm_101/while/lstm_cell_101/dropout_1/random_uniform/RandomUniformГ
5lstm_101/while/lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>27
5lstm_101/while/lstm_cell_101/dropout_1/GreaterEqual/yК
3lstm_101/while/lstm_cell_101/dropout_1/GreaterEqualGreaterEqualLlstm_101/while/lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:0>lstm_101/while/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3lstm_101/while/lstm_cell_101/dropout_1/GreaterEqualм
+lstm_101/while/lstm_cell_101/dropout_1/CastCast7lstm_101/while/lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_101/while/lstm_cell_101/dropout_1/Castі
,lstm_101/while/lstm_cell_101/dropout_1/Mul_1Mul.lstm_101/while/lstm_cell_101/dropout_1/Mul:z:0/lstm_101/while/lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,lstm_101/while/lstm_cell_101/dropout_1/Mul_1Ё
,lstm_101/while/lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2.
,lstm_101/while/lstm_cell_101/dropout_2/Constљ
*lstm_101/while/lstm_cell_101/dropout_2/MulMul/lstm_101/while/lstm_cell_101/ones_like:output:05lstm_101/while/lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_101/while/lstm_cell_101/dropout_2/MulЛ
,lstm_101/while/lstm_cell_101/dropout_2/ShapeShape/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_101/while/lstm_cell_101/dropout_2/ShapeЎ
Clstm_101/while/lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform5lstm_101/while/lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2СЌь2E
Clstm_101/while/lstm_cell_101/dropout_2/random_uniform/RandomUniformГ
5lstm_101/while/lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>27
5lstm_101/while/lstm_cell_101/dropout_2/GreaterEqual/yК
3lstm_101/while/lstm_cell_101/dropout_2/GreaterEqualGreaterEqualLlstm_101/while/lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:0>lstm_101/while/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3lstm_101/while/lstm_cell_101/dropout_2/GreaterEqualм
+lstm_101/while/lstm_cell_101/dropout_2/CastCast7lstm_101/while/lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_101/while/lstm_cell_101/dropout_2/Castі
,lstm_101/while/lstm_cell_101/dropout_2/Mul_1Mul.lstm_101/while/lstm_cell_101/dropout_2/Mul:z:0/lstm_101/while/lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,lstm_101/while/lstm_cell_101/dropout_2/Mul_1Ё
,lstm_101/while/lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2.
,lstm_101/while/lstm_cell_101/dropout_3/Constљ
*lstm_101/while/lstm_cell_101/dropout_3/MulMul/lstm_101/while/lstm_cell_101/ones_like:output:05lstm_101/while/lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_101/while/lstm_cell_101/dropout_3/MulЛ
,lstm_101/while/lstm_cell_101/dropout_3/ShapeShape/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2.
,lstm_101/while/lstm_cell_101/dropout_3/Shape­
Clstm_101/while/lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform5lstm_101/while/lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ф!2E
Clstm_101/while/lstm_cell_101/dropout_3/random_uniform/RandomUniformГ
5lstm_101/while/lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>27
5lstm_101/while/lstm_cell_101/dropout_3/GreaterEqual/yК
3lstm_101/while/lstm_cell_101/dropout_3/GreaterEqualGreaterEqualLlstm_101/while/lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:0>lstm_101/while/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3lstm_101/while/lstm_cell_101/dropout_3/GreaterEqualм
+lstm_101/while/lstm_cell_101/dropout_3/CastCast7lstm_101/while/lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_101/while/lstm_cell_101/dropout_3/Castі
,lstm_101/while/lstm_cell_101/dropout_3/Mul_1Mul.lstm_101/while/lstm_cell_101/dropout_3/Mul:z:0/lstm_101/while/lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,lstm_101/while/lstm_cell_101/dropout_3/Mul_1
,lstm_101/while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_101/while/lstm_cell_101/split/split_dimф
1lstm_101/while/lstm_cell_101/split/ReadVariableOpReadVariableOp<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1lstm_101/while/lstm_cell_101/split/ReadVariableOp
"lstm_101/while/lstm_cell_101/splitSplit5lstm_101/while/lstm_cell_101/split/split_dim:output:09lstm_101/while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2$
"lstm_101/while/lstm_cell_101/splitю
#lstm_101/while/lstm_cell_101/MatMulMatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_101/while/lstm_cell_101/MatMulђ
%lstm_101/while/lstm_cell_101/MatMul_1MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_1ђ
%lstm_101/while/lstm_cell_101/MatMul_2MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_2ђ
%lstm_101/while/lstm_cell_101/MatMul_3MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_3Ђ
.lstm_101/while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.lstm_101/while/lstm_cell_101/split_1/split_dimц
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype025
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp
$lstm_101/while/lstm_cell_101/split_1Split7lstm_101/while/lstm_cell_101/split_1/split_dim:output:0;lstm_101/while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2&
$lstm_101/while/lstm_cell_101/split_1ч
$lstm_101/while/lstm_cell_101/BiasAddBiasAdd-lstm_101/while/lstm_cell_101/MatMul:product:0-lstm_101/while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/while/lstm_cell_101/BiasAddэ
&lstm_101/while/lstm_cell_101/BiasAdd_1BiasAdd/lstm_101/while/lstm_cell_101/MatMul_1:product:0-lstm_101/while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_1э
&lstm_101/while/lstm_cell_101/BiasAdd_2BiasAdd/lstm_101/while/lstm_cell_101/MatMul_2:product:0-lstm_101/while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_2э
&lstm_101/while/lstm_cell_101/BiasAdd_3BiasAdd/lstm_101/while/lstm_cell_101/MatMul_3:product:0-lstm_101/while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_3Ы
 lstm_101/while/lstm_cell_101/mulMullstm_101_while_placeholder_2.lstm_101/while/lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/while/lstm_cell_101/mulб
"lstm_101/while/lstm_cell_101/mul_1Mullstm_101_while_placeholder_20lstm_101/while/lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_1б
"lstm_101/while/lstm_cell_101/mul_2Mullstm_101_while_placeholder_20lstm_101/while/lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_2б
"lstm_101/while/lstm_cell_101/mul_3Mullstm_101_while_placeholder_20lstm_101/while/lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_3в
+lstm_101/while/lstm_cell_101/ReadVariableOpReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_101/while/lstm_cell_101/ReadVariableOpЕ
0lstm_101/while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_101/while/lstm_cell_101/strided_slice/stackЙ
2lstm_101/while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_101/while/lstm_cell_101/strided_slice/stack_1Й
2lstm_101/while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_101/while/lstm_cell_101/strided_slice/stack_2Њ
*lstm_101/while/lstm_cell_101/strided_sliceStridedSlice3lstm_101/while/lstm_cell_101/ReadVariableOp:value:09lstm_101/while/lstm_cell_101/strided_slice/stack:output:0;lstm_101/while/lstm_cell_101/strided_slice/stack_1:output:0;lstm_101/while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_101/while/lstm_cell_101/strided_sliceх
%lstm_101/while/lstm_cell_101/MatMul_4MatMul$lstm_101/while/lstm_cell_101/mul:z:03lstm_101/while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_4п
 lstm_101/while/lstm_cell_101/addAddV2-lstm_101/while/lstm_cell_101/BiasAdd:output:0/lstm_101/while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/while/lstm_cell_101/addЏ
$lstm_101/while/lstm_cell_101/SigmoidSigmoid$lstm_101/while/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/while/lstm_cell_101/Sigmoidж
-lstm_101/while/lstm_cell_101/ReadVariableOp_1ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_1Й
2lstm_101/while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_101/while/lstm_cell_101/strided_slice_1/stackН
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   26
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_1StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_1:value:0;lstm_101/while/lstm_cell_101/strided_slice_1/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_1/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_1щ
%lstm_101/while/lstm_cell_101/MatMul_5MatMul&lstm_101/while/lstm_cell_101/mul_1:z:05lstm_101/while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_5х
"lstm_101/while/lstm_cell_101/add_1AddV2/lstm_101/while/lstm_cell_101/BiasAdd_1:output:0/lstm_101/while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_1Е
&lstm_101/while/lstm_cell_101/Sigmoid_1Sigmoid&lstm_101/while/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/Sigmoid_1Ы
"lstm_101/while/lstm_cell_101/mul_4Mul*lstm_101/while/lstm_cell_101/Sigmoid_1:y:0lstm_101_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_4ж
-lstm_101/while/lstm_cell_101/ReadVariableOp_2ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_2Й
2lstm_101/while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_101/while/lstm_cell_101/strided_slice_2/stackН
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   26
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_2StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_2:value:0;lstm_101/while/lstm_cell_101/strided_slice_2/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_2/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_2щ
%lstm_101/while/lstm_cell_101/MatMul_6MatMul&lstm_101/while/lstm_cell_101/mul_2:z:05lstm_101/while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_6х
"lstm_101/while/lstm_cell_101/add_2AddV2/lstm_101/while/lstm_cell_101/BiasAdd_2:output:0/lstm_101/while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_2Ј
!lstm_101/while/lstm_cell_101/ReluRelu&lstm_101/while/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_101/while/lstm_cell_101/Reluм
"lstm_101/while/lstm_cell_101/mul_5Mul(lstm_101/while/lstm_cell_101/Sigmoid:y:0/lstm_101/while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_5г
"lstm_101/while/lstm_cell_101/add_3AddV2&lstm_101/while/lstm_cell_101/mul_4:z:0&lstm_101/while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_3ж
-lstm_101/while/lstm_cell_101/ReadVariableOp_3ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_3Й
2lstm_101/while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_101/while/lstm_cell_101/strided_slice_3/stackН
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_3StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_3:value:0;lstm_101/while/lstm_cell_101/strided_slice_3/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_3/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_3щ
%lstm_101/while/lstm_cell_101/MatMul_7MatMul&lstm_101/while/lstm_cell_101/mul_3:z:05lstm_101/while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_7х
"lstm_101/while/lstm_cell_101/add_4AddV2/lstm_101/while/lstm_cell_101/BiasAdd_3:output:0/lstm_101/while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_4Е
&lstm_101/while/lstm_cell_101/Sigmoid_2Sigmoid&lstm_101/while/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/Sigmoid_2Ќ
#lstm_101/while/lstm_cell_101/Relu_1Relu&lstm_101/while/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_101/while/lstm_cell_101/Relu_1р
"lstm_101/while/lstm_cell_101/mul_6Mul*lstm_101/while/lstm_cell_101/Sigmoid_2:y:01lstm_101/while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_6
3lstm_101/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_101_while_placeholder_1lstm_101_while_placeholder&lstm_101/while/lstm_cell_101/mul_6:z:0*
_output_shapes
: *
element_dtype025
3lstm_101/while/TensorArrayV2Write/TensorListSetItemn
lstm_101/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_101/while/add/y
lstm_101/while/addAddV2lstm_101_while_placeholderlstm_101/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_101/while/addr
lstm_101/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_101/while/add_1/yЃ
lstm_101/while/add_1AddV2*lstm_101_while_lstm_101_while_loop_counterlstm_101/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_101/while/add_1
lstm_101/while/IdentityIdentitylstm_101/while/add_1:z:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/IdentityЋ
lstm_101/while/Identity_1Identity0lstm_101_while_lstm_101_while_maximum_iterations^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_1
lstm_101/while/Identity_2Identitylstm_101/while/add:z:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_2О
lstm_101/while/Identity_3IdentityClstm_101/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_3В
lstm_101/while/Identity_4Identity&lstm_101/while/lstm_cell_101/mul_6:z:0^lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/while/Identity_4В
lstm_101/while/Identity_5Identity&lstm_101/while/lstm_cell_101/add_3:z:0^lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/while/Identity_5
lstm_101/while/NoOpNoOp,^lstm_101/while/lstm_cell_101/ReadVariableOp.^lstm_101/while/lstm_cell_101/ReadVariableOp_1.^lstm_101/while/lstm_cell_101/ReadVariableOp_2.^lstm_101/while/lstm_cell_101/ReadVariableOp_32^lstm_101/while/lstm_cell_101/split/ReadVariableOp4^lstm_101/while/lstm_cell_101/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_101/while/NoOp";
lstm_101_while_identity lstm_101/while/Identity:output:0"?
lstm_101_while_identity_1"lstm_101/while/Identity_1:output:0"?
lstm_101_while_identity_2"lstm_101/while/Identity_2:output:0"?
lstm_101_while_identity_3"lstm_101/while/Identity_3:output:0"?
lstm_101_while_identity_4"lstm_101/while/Identity_4:output:0"?
lstm_101_while_identity_5"lstm_101/while/Identity_5:output:0"T
'lstm_101_while_lstm_101_strided_slice_1)lstm_101_while_lstm_101_strided_slice_1_0"n
4lstm_101_while_lstm_cell_101_readvariableop_resource6lstm_101_while_lstm_cell_101_readvariableop_resource_0"~
<lstm_101_while_lstm_cell_101_split_1_readvariableop_resource>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0"z
:lstm_101_while_lstm_cell_101_split_readvariableop_resource<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0"Ь
clstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensorelstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2Z
+lstm_101/while/lstm_cell_101/ReadVariableOp+lstm_101/while/lstm_cell_101/ReadVariableOp2^
-lstm_101/while/lstm_cell_101/ReadVariableOp_1-lstm_101/while/lstm_cell_101/ReadVariableOp_12^
-lstm_101/while/lstm_cell_101/ReadVariableOp_2-lstm_101/while/lstm_cell_101/ReadVariableOp_22^
-lstm_101/while/lstm_cell_101/ReadVariableOp_3-lstm_101/while/lstm_cell_101/ReadVariableOp_32f
1lstm_101/while/lstm_cell_101/split/ReadVariableOp1lstm_101/while/lstm_cell_101/split/ReadVariableOp2j
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp: 
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

Я
__inference_loss_fn_1_3241482[
Hlstm_101_lstm_cell_101_kernel_regularizer_square_readvariableop_resource:	
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHlstm_101_lstm_cell_101_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mul{
IdentityIdentity1lstm_101/lstm_cell_101/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp
У
	
"__inference__wrapped_model_3237685
input_42U
Bsequential_41_lstm_101_lstm_cell_101_split_readvariableop_resource:	S
Dsequential_41_lstm_101_lstm_cell_101_split_1_readvariableop_resource:	O
<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource:	 H
6sequential_41_dense_122_matmul_readvariableop_resource:  E
7sequential_41_dense_122_biasadd_readvariableop_resource: H
6sequential_41_dense_123_matmul_readvariableop_resource: E
7sequential_41_dense_123_biasadd_readvariableop_resource:
identityЂ.sequential_41/dense_122/BiasAdd/ReadVariableOpЂ-sequential_41/dense_122/MatMul/ReadVariableOpЂ.sequential_41/dense_123/BiasAdd/ReadVariableOpЂ-sequential_41/dense_123/MatMul/ReadVariableOpЂ3sequential_41/lstm_101/lstm_cell_101/ReadVariableOpЂ5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_1Ђ5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_2Ђ5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_3Ђ9sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOpЂ;sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOpЂsequential_41/lstm_101/whilet
sequential_41/lstm_101/ShapeShapeinput_42*
T0*
_output_shapes
:2
sequential_41/lstm_101/ShapeЂ
*sequential_41/lstm_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_41/lstm_101/strided_slice/stackІ
,sequential_41/lstm_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_41/lstm_101/strided_slice/stack_1І
,sequential_41/lstm_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_41/lstm_101/strided_slice/stack_2ь
$sequential_41/lstm_101/strided_sliceStridedSlice%sequential_41/lstm_101/Shape:output:03sequential_41/lstm_101/strided_slice/stack:output:05sequential_41/lstm_101/strided_slice/stack_1:output:05sequential_41/lstm_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_41/lstm_101/strided_slice
"sequential_41/lstm_101/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_41/lstm_101/zeros/mul/yШ
 sequential_41/lstm_101/zeros/mulMul-sequential_41/lstm_101/strided_slice:output:0+sequential_41/lstm_101/zeros/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_41/lstm_101/zeros/mul
#sequential_41/lstm_101/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_41/lstm_101/zeros/Less/yУ
!sequential_41/lstm_101/zeros/LessLess$sequential_41/lstm_101/zeros/mul:z:0,sequential_41/lstm_101/zeros/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_41/lstm_101/zeros/Less
%sequential_41/lstm_101/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_41/lstm_101/zeros/packed/1п
#sequential_41/lstm_101/zeros/packedPack-sequential_41/lstm_101/strided_slice:output:0.sequential_41/lstm_101/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_41/lstm_101/zeros/packed
"sequential_41/lstm_101/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_41/lstm_101/zeros/Constб
sequential_41/lstm_101/zerosFill,sequential_41/lstm_101/zeros/packed:output:0+sequential_41/lstm_101/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_41/lstm_101/zeros
$sequential_41/lstm_101/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_41/lstm_101/zeros_1/mul/yЮ
"sequential_41/lstm_101/zeros_1/mulMul-sequential_41/lstm_101/strided_slice:output:0-sequential_41/lstm_101/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2$
"sequential_41/lstm_101/zeros_1/mul
%sequential_41/lstm_101/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2'
%sequential_41/lstm_101/zeros_1/Less/yЫ
#sequential_41/lstm_101/zeros_1/LessLess&sequential_41/lstm_101/zeros_1/mul:z:0.sequential_41/lstm_101/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2%
#sequential_41/lstm_101/zeros_1/Less
'sequential_41/lstm_101/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_41/lstm_101/zeros_1/packed/1х
%sequential_41/lstm_101/zeros_1/packedPack-sequential_41/lstm_101/strided_slice:output:00sequential_41/lstm_101/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_41/lstm_101/zeros_1/packed
$sequential_41/lstm_101/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$sequential_41/lstm_101/zeros_1/Constй
sequential_41/lstm_101/zeros_1Fill.sequential_41/lstm_101/zeros_1/packed:output:0-sequential_41/lstm_101/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
sequential_41/lstm_101/zeros_1Ѓ
%sequential_41/lstm_101/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_41/lstm_101/transpose/permС
 sequential_41/lstm_101/transpose	Transposeinput_42.sequential_41/lstm_101/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_41/lstm_101/transpose
sequential_41/lstm_101/Shape_1Shape$sequential_41/lstm_101/transpose:y:0*
T0*
_output_shapes
:2 
sequential_41/lstm_101/Shape_1І
,sequential_41/lstm_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_41/lstm_101/strided_slice_1/stackЊ
.sequential_41/lstm_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/lstm_101/strided_slice_1/stack_1Њ
.sequential_41/lstm_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/lstm_101/strided_slice_1/stack_2ј
&sequential_41/lstm_101/strided_slice_1StridedSlice'sequential_41/lstm_101/Shape_1:output:05sequential_41/lstm_101/strided_slice_1/stack:output:07sequential_41/lstm_101/strided_slice_1/stack_1:output:07sequential_41/lstm_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_41/lstm_101/strided_slice_1Г
2sequential_41/lstm_101/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2sequential_41/lstm_101/TensorArrayV2/element_shape
$sequential_41/lstm_101/TensorArrayV2TensorListReserve;sequential_41/lstm_101/TensorArrayV2/element_shape:output:0/sequential_41/lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_41/lstm_101/TensorArrayV2э
Lsequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2N
Lsequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shapeд
>sequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_41/lstm_101/transpose:y:0Usequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensorІ
,sequential_41/lstm_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_41/lstm_101/strided_slice_2/stackЊ
.sequential_41/lstm_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/lstm_101/strided_slice_2/stack_1Њ
.sequential_41/lstm_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/lstm_101/strided_slice_2/stack_2
&sequential_41/lstm_101/strided_slice_2StridedSlice$sequential_41/lstm_101/transpose:y:05sequential_41/lstm_101/strided_slice_2/stack:output:07sequential_41/lstm_101/strided_slice_2/stack_1:output:07sequential_41/lstm_101/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2(
&sequential_41/lstm_101/strided_slice_2С
4sequential_41/lstm_101/lstm_cell_101/ones_like/ShapeShape%sequential_41/lstm_101/zeros:output:0*
T0*
_output_shapes
:26
4sequential_41/lstm_101/lstm_cell_101/ones_like/ShapeБ
4sequential_41/lstm_101/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_41/lstm_101/lstm_cell_101/ones_like/Const
.sequential_41/lstm_101/lstm_cell_101/ones_likeFill=sequential_41/lstm_101/lstm_cell_101/ones_like/Shape:output:0=sequential_41/lstm_101/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/ones_likeЎ
4sequential_41/lstm_101/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_41/lstm_101/lstm_cell_101/split/split_dimњ
9sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOpReadVariableOpBsequential_41_lstm_101_lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02;
9sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOpЛ
*sequential_41/lstm_101/lstm_cell_101/splitSplit=sequential_41/lstm_101/lstm_cell_101/split/split_dim:output:0Asequential_41/lstm_101/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2,
*sequential_41/lstm_101/lstm_cell_101/splitќ
+sequential_41/lstm_101/lstm_cell_101/MatMulMatMul/sequential_41/lstm_101/strided_slice_2:output:03sequential_41/lstm_101/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_41/lstm_101/lstm_cell_101/MatMul
-sequential_41/lstm_101/lstm_cell_101/MatMul_1MatMul/sequential_41/lstm_101/strided_slice_2:output:03sequential_41/lstm_101/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_1
-sequential_41/lstm_101/lstm_cell_101/MatMul_2MatMul/sequential_41/lstm_101/strided_slice_2:output:03sequential_41/lstm_101/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_2
-sequential_41/lstm_101/lstm_cell_101/MatMul_3MatMul/sequential_41/lstm_101/strided_slice_2:output:03sequential_41/lstm_101/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_3В
6sequential_41/lstm_101/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_41/lstm_101/lstm_cell_101/split_1/split_dimќ
;sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOpReadVariableOpDsequential_41_lstm_101_lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02=
;sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOpГ
,sequential_41/lstm_101/lstm_cell_101/split_1Split?sequential_41/lstm_101/lstm_cell_101/split_1/split_dim:output:0Csequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2.
,sequential_41/lstm_101/lstm_cell_101/split_1
,sequential_41/lstm_101/lstm_cell_101/BiasAddBiasAdd5sequential_41/lstm_101/lstm_cell_101/MatMul:product:05sequential_41/lstm_101/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_41/lstm_101/lstm_cell_101/BiasAdd
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_1BiasAdd7sequential_41/lstm_101/lstm_cell_101/MatMul_1:product:05sequential_41/lstm_101/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_1
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_2BiasAdd7sequential_41/lstm_101/lstm_cell_101/MatMul_2:product:05sequential_41/lstm_101/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_2
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_3BiasAdd7sequential_41/lstm_101/lstm_cell_101/MatMul_3:product:05sequential_41/lstm_101/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/BiasAdd_3э
(sequential_41/lstm_101/lstm_cell_101/mulMul%sequential_41/lstm_101/zeros:output:07sequential_41/lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_41/lstm_101/lstm_cell_101/mulё
*sequential_41/lstm_101/lstm_cell_101/mul_1Mul%sequential_41/lstm_101/zeros:output:07sequential_41/lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_1ё
*sequential_41/lstm_101/lstm_cell_101/mul_2Mul%sequential_41/lstm_101/zeros:output:07sequential_41/lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_2ё
*sequential_41/lstm_101/lstm_cell_101/mul_3Mul%sequential_41/lstm_101/zeros:output:07sequential_41/lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_3ш
3sequential_41/lstm_101/lstm_cell_101/ReadVariableOpReadVariableOp<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype025
3sequential_41/lstm_101/lstm_cell_101/ReadVariableOpХ
8sequential_41/lstm_101/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_41/lstm_101/lstm_cell_101/strided_slice/stackЩ
:sequential_41/lstm_101/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_41/lstm_101/lstm_cell_101/strided_slice/stack_1Щ
:sequential_41/lstm_101/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_41/lstm_101/lstm_cell_101/strided_slice/stack_2к
2sequential_41/lstm_101/lstm_cell_101/strided_sliceStridedSlice;sequential_41/lstm_101/lstm_cell_101/ReadVariableOp:value:0Asequential_41/lstm_101/lstm_cell_101/strided_slice/stack:output:0Csequential_41/lstm_101/lstm_cell_101/strided_slice/stack_1:output:0Csequential_41/lstm_101/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2sequential_41/lstm_101/lstm_cell_101/strided_slice
-sequential_41/lstm_101/lstm_cell_101/MatMul_4MatMul,sequential_41/lstm_101/lstm_cell_101/mul:z:0;sequential_41/lstm_101/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_4џ
(sequential_41/lstm_101/lstm_cell_101/addAddV25sequential_41/lstm_101/lstm_cell_101/BiasAdd:output:07sequential_41/lstm_101/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_41/lstm_101/lstm_cell_101/addЧ
,sequential_41/lstm_101/lstm_cell_101/SigmoidSigmoid,sequential_41/lstm_101/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_41/lstm_101/lstm_cell_101/Sigmoidь
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_1ReadVariableOp<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype027
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_1Щ
:sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stackЭ
<sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_1Э
<sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_2ц
4sequential_41/lstm_101/lstm_cell_101/strided_slice_1StridedSlice=sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_1:value:0Csequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_1:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_41/lstm_101/lstm_cell_101/strided_slice_1
-sequential_41/lstm_101/lstm_cell_101/MatMul_5MatMul.sequential_41/lstm_101/lstm_cell_101/mul_1:z:0=sequential_41/lstm_101/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_5
*sequential_41/lstm_101/lstm_cell_101/add_1AddV27sequential_41/lstm_101/lstm_cell_101/BiasAdd_1:output:07sequential_41/lstm_101/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/add_1Э
.sequential_41/lstm_101/lstm_cell_101/Sigmoid_1Sigmoid.sequential_41/lstm_101/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/Sigmoid_1ю
*sequential_41/lstm_101/lstm_cell_101/mul_4Mul2sequential_41/lstm_101/lstm_cell_101/Sigmoid_1:y:0'sequential_41/lstm_101/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_4ь
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_2ReadVariableOp<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype027
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_2Щ
:sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2<
:sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stackЭ
<sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_1Э
<sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_2ц
4sequential_41/lstm_101/lstm_cell_101/strided_slice_2StridedSlice=sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_2:value:0Csequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_1:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_41/lstm_101/lstm_cell_101/strided_slice_2
-sequential_41/lstm_101/lstm_cell_101/MatMul_6MatMul.sequential_41/lstm_101/lstm_cell_101/mul_2:z:0=sequential_41/lstm_101/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_6
*sequential_41/lstm_101/lstm_cell_101/add_2AddV27sequential_41/lstm_101/lstm_cell_101/BiasAdd_2:output:07sequential_41/lstm_101/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/add_2Р
)sequential_41/lstm_101/lstm_cell_101/ReluRelu.sequential_41/lstm_101/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_41/lstm_101/lstm_cell_101/Reluќ
*sequential_41/lstm_101/lstm_cell_101/mul_5Mul0sequential_41/lstm_101/lstm_cell_101/Sigmoid:y:07sequential_41/lstm_101/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_5ѓ
*sequential_41/lstm_101/lstm_cell_101/add_3AddV2.sequential_41/lstm_101/lstm_cell_101/mul_4:z:0.sequential_41/lstm_101/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/add_3ь
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_3ReadVariableOp<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype027
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_3Щ
:sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2<
:sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stackЭ
<sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_1Э
<sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_2ц
4sequential_41/lstm_101/lstm_cell_101/strided_slice_3StridedSlice=sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_3:value:0Csequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_1:output:0Esequential_41/lstm_101/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_41/lstm_101/lstm_cell_101/strided_slice_3
-sequential_41/lstm_101/lstm_cell_101/MatMul_7MatMul.sequential_41/lstm_101/lstm_cell_101/mul_3:z:0=sequential_41/lstm_101/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_41/lstm_101/lstm_cell_101/MatMul_7
*sequential_41/lstm_101/lstm_cell_101/add_4AddV27sequential_41/lstm_101/lstm_cell_101/BiasAdd_3:output:07sequential_41/lstm_101/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/add_4Э
.sequential_41/lstm_101/lstm_cell_101/Sigmoid_2Sigmoid.sequential_41/lstm_101/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/lstm_cell_101/Sigmoid_2Ф
+sequential_41/lstm_101/lstm_cell_101/Relu_1Relu.sequential_41/lstm_101/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_41/lstm_101/lstm_cell_101/Relu_1
*sequential_41/lstm_101/lstm_cell_101/mul_6Mul2sequential_41/lstm_101/lstm_cell_101/Sigmoid_2:y:09sequential_41/lstm_101/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_41/lstm_101/lstm_cell_101/mul_6Н
4sequential_41/lstm_101/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4sequential_41/lstm_101/TensorArrayV2_1/element_shape
&sequential_41/lstm_101/TensorArrayV2_1TensorListReserve=sequential_41/lstm_101/TensorArrayV2_1/element_shape:output:0/sequential_41/lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&sequential_41/lstm_101/TensorArrayV2_1|
sequential_41/lstm_101/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_41/lstm_101/time­
/sequential_41/lstm_101/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ21
/sequential_41/lstm_101/while/maximum_iterations
)sequential_41/lstm_101/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_41/lstm_101/while/loop_counterп
sequential_41/lstm_101/whileWhile2sequential_41/lstm_101/while/loop_counter:output:08sequential_41/lstm_101/while/maximum_iterations:output:0$sequential_41/lstm_101/time:output:0/sequential_41/lstm_101/TensorArrayV2_1:handle:0%sequential_41/lstm_101/zeros:output:0'sequential_41/lstm_101/zeros_1:output:0/sequential_41/lstm_101/strided_slice_1:output:0Nsequential_41/lstm_101/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bsequential_41_lstm_101_lstm_cell_101_split_readvariableop_resourceDsequential_41_lstm_101_lstm_cell_101_split_1_readvariableop_resource<sequential_41_lstm_101_lstm_cell_101_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_41_lstm_101_while_body_3237536*5
cond-R+
)sequential_41_lstm_101_while_cond_3237535*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_41/lstm_101/whileу
Gsequential_41/lstm_101/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2I
Gsequential_41/lstm_101/TensorArrayV2Stack/TensorListStack/element_shapeФ
9sequential_41/lstm_101/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_41/lstm_101/while:output:3Psequential_41/lstm_101/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02;
9sequential_41/lstm_101/TensorArrayV2Stack/TensorListStackЏ
,sequential_41/lstm_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2.
,sequential_41/lstm_101/strided_slice_3/stackЊ
.sequential_41/lstm_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_41/lstm_101/strided_slice_3/stack_1Њ
.sequential_41/lstm_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/lstm_101/strided_slice_3/stack_2Є
&sequential_41/lstm_101/strided_slice_3StridedSliceBsequential_41/lstm_101/TensorArrayV2Stack/TensorListStack:tensor:05sequential_41/lstm_101/strided_slice_3/stack:output:07sequential_41/lstm_101/strided_slice_3/stack_1:output:07sequential_41/lstm_101/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2(
&sequential_41/lstm_101/strided_slice_3Ї
'sequential_41/lstm_101/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'sequential_41/lstm_101/transpose_1/perm
"sequential_41/lstm_101/transpose_1	TransposeBsequential_41/lstm_101/TensorArrayV2Stack/TensorListStack:tensor:00sequential_41/lstm_101/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2$
"sequential_41/lstm_101/transpose_1
sequential_41/lstm_101/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_41/lstm_101/runtimeе
-sequential_41/dense_122/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_41/dense_122/MatMul/ReadVariableOpф
sequential_41/dense_122/MatMulMatMul/sequential_41/lstm_101/strided_slice_3:output:05sequential_41/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
sequential_41/dense_122/MatMulд
.sequential_41/dense_122/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_41/dense_122/BiasAdd/ReadVariableOpс
sequential_41/dense_122/BiasAddBiasAdd(sequential_41/dense_122/MatMul:product:06sequential_41/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
sequential_41/dense_122/BiasAdd 
sequential_41/dense_122/ReluRelu(sequential_41/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_41/dense_122/Reluе
-sequential_41/dense_123/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-sequential_41/dense_123/MatMul/ReadVariableOpп
sequential_41/dense_123/MatMulMatMul*sequential_41/dense_122/Relu:activations:05sequential_41/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_41/dense_123/MatMulд
.sequential_41/dense_123/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_41/dense_123/BiasAdd/ReadVariableOpс
sequential_41/dense_123/BiasAddBiasAdd(sequential_41/dense_123/MatMul:product:06sequential_41/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_41/dense_123/BiasAdd
sequential_41/reshape_61/ShapeShape(sequential_41/dense_123/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_41/reshape_61/ShapeІ
,sequential_41/reshape_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_41/reshape_61/strided_slice/stackЊ
.sequential_41/reshape_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/reshape_61/strided_slice/stack_1Њ
.sequential_41/reshape_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_41/reshape_61/strided_slice/stack_2ј
&sequential_41/reshape_61/strided_sliceStridedSlice'sequential_41/reshape_61/Shape:output:05sequential_41/reshape_61/strided_slice/stack:output:07sequential_41/reshape_61/strided_slice/stack_1:output:07sequential_41/reshape_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_41/reshape_61/strided_slice
(sequential_41/reshape_61/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_41/reshape_61/Reshape/shape/1
(sequential_41/reshape_61/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_41/reshape_61/Reshape/shape/2
&sequential_41/reshape_61/Reshape/shapePack/sequential_41/reshape_61/strided_slice:output:01sequential_41/reshape_61/Reshape/shape/1:output:01sequential_41/reshape_61/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_41/reshape_61/Reshape/shapeр
 sequential_41/reshape_61/ReshapeReshape(sequential_41/dense_123/BiasAdd:output:0/sequential_41/reshape_61/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_41/reshape_61/Reshape
IdentityIdentity)sequential_41/reshape_61/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp/^sequential_41/dense_122/BiasAdd/ReadVariableOp.^sequential_41/dense_122/MatMul/ReadVariableOp/^sequential_41/dense_123/BiasAdd/ReadVariableOp.^sequential_41/dense_123/MatMul/ReadVariableOp4^sequential_41/lstm_101/lstm_cell_101/ReadVariableOp6^sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_16^sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_26^sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_3:^sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOp<^sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOp^sequential_41/lstm_101/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2`
.sequential_41/dense_122/BiasAdd/ReadVariableOp.sequential_41/dense_122/BiasAdd/ReadVariableOp2^
-sequential_41/dense_122/MatMul/ReadVariableOp-sequential_41/dense_122/MatMul/ReadVariableOp2`
.sequential_41/dense_123/BiasAdd/ReadVariableOp.sequential_41/dense_123/BiasAdd/ReadVariableOp2^
-sequential_41/dense_123/MatMul/ReadVariableOp-sequential_41/dense_123/MatMul/ReadVariableOp2j
3sequential_41/lstm_101/lstm_cell_101/ReadVariableOp3sequential_41/lstm_101/lstm_cell_101/ReadVariableOp2n
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_15sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_12n
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_25sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_22n
5sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_35sequential_41/lstm_101/lstm_cell_101/ReadVariableOp_32v
9sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOp9sequential_41/lstm_101/lstm_cell_101/split/ReadVariableOp2z
;sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOp;sequential_41/lstm_101/lstm_cell_101/split_1/ReadVariableOp2<
sequential_41/lstm_101/whilesequential_41/lstm_101/while:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_42
УН
й
)sequential_41_lstm_101_while_body_3237536J
Fsequential_41_lstm_101_while_sequential_41_lstm_101_while_loop_counterP
Lsequential_41_lstm_101_while_sequential_41_lstm_101_while_maximum_iterations,
(sequential_41_lstm_101_while_placeholder.
*sequential_41_lstm_101_while_placeholder_1.
*sequential_41_lstm_101_while_placeholder_2.
*sequential_41_lstm_101_while_placeholder_3I
Esequential_41_lstm_101_while_sequential_41_lstm_101_strided_slice_1_0
sequential_41_lstm_101_while_tensorarrayv2read_tensorlistgetitem_sequential_41_lstm_101_tensorarrayunstack_tensorlistfromtensor_0]
Jsequential_41_lstm_101_while_lstm_cell_101_split_readvariableop_resource_0:	[
Lsequential_41_lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0:	W
Dsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0:	 )
%sequential_41_lstm_101_while_identity+
'sequential_41_lstm_101_while_identity_1+
'sequential_41_lstm_101_while_identity_2+
'sequential_41_lstm_101_while_identity_3+
'sequential_41_lstm_101_while_identity_4+
'sequential_41_lstm_101_while_identity_5G
Csequential_41_lstm_101_while_sequential_41_lstm_101_strided_slice_1
sequential_41_lstm_101_while_tensorarrayv2read_tensorlistgetitem_sequential_41_lstm_101_tensorarrayunstack_tensorlistfromtensor[
Hsequential_41_lstm_101_while_lstm_cell_101_split_readvariableop_resource:	Y
Jsequential_41_lstm_101_while_lstm_cell_101_split_1_readvariableop_resource:	U
Bsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource:	 Ђ9sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOpЂ;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1Ђ;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2Ђ;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3Ђ?sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOpЂAsequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOpё
Nsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2P
Nsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shapeо
@sequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_41_lstm_101_while_tensorarrayv2read_tensorlistgetitem_sequential_41_lstm_101_tensorarrayunstack_tensorlistfromtensor_0(sequential_41_lstm_101_while_placeholderWsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02B
@sequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItemв
:sequential_41/lstm_101/while/lstm_cell_101/ones_like/ShapeShape*sequential_41_lstm_101_while_placeholder_2*
T0*
_output_shapes
:2<
:sequential_41/lstm_101/while/lstm_cell_101/ones_like/ShapeН
:sequential_41/lstm_101/while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:sequential_41/lstm_101/while/lstm_cell_101/ones_like/ConstА
4sequential_41/lstm_101/while/lstm_cell_101/ones_likeFillCsequential_41/lstm_101/while/lstm_cell_101/ones_like/Shape:output:0Csequential_41/lstm_101/while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/ones_likeК
:sequential_41/lstm_101/while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_41/lstm_101/while/lstm_cell_101/split/split_dim
?sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOpReadVariableOpJsequential_41_lstm_101_while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02A
?sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOpг
0sequential_41/lstm_101/while/lstm_cell_101/splitSplitCsequential_41/lstm_101/while/lstm_cell_101/split/split_dim:output:0Gsequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split22
0sequential_41/lstm_101/while/lstm_cell_101/splitІ
1sequential_41/lstm_101/while/lstm_cell_101/MatMulMatMulGsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_41/lstm_101/while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_41/lstm_101/while/lstm_cell_101/MatMulЊ
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_1MatMulGsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_41/lstm_101/while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_1Њ
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_2MatMulGsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_41/lstm_101/while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_2Њ
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_3MatMulGsequential_41/lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_41/lstm_101/while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_3О
<sequential_41/lstm_101/while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_41/lstm_101/while/lstm_cell_101/split_1/split_dim
Asequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOpReadVariableOpLsequential_41_lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02C
Asequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOpЫ
2sequential_41/lstm_101/while/lstm_cell_101/split_1SplitEsequential_41/lstm_101/while/lstm_cell_101/split_1/split_dim:output:0Isequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split24
2sequential_41/lstm_101/while/lstm_cell_101/split_1
2sequential_41/lstm_101/while/lstm_cell_101/BiasAddBiasAdd;sequential_41/lstm_101/while/lstm_cell_101/MatMul:product:0;sequential_41/lstm_101/while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_41/lstm_101/while/lstm_cell_101/BiasAddЅ
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_1BiasAdd=sequential_41/lstm_101/while/lstm_cell_101/MatMul_1:product:0;sequential_41/lstm_101/while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_1Ѕ
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_2BiasAdd=sequential_41/lstm_101/while/lstm_cell_101/MatMul_2:product:0;sequential_41/lstm_101/while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_2Ѕ
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_3BiasAdd=sequential_41/lstm_101/while/lstm_cell_101/MatMul_3:product:0;sequential_41/lstm_101/while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_3
.sequential_41/lstm_101/while/lstm_cell_101/mulMul*sequential_41_lstm_101_while_placeholder_2=sequential_41/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/while/lstm_cell_101/mul
0sequential_41/lstm_101/while/lstm_cell_101/mul_1Mul*sequential_41_lstm_101_while_placeholder_2=sequential_41/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_1
0sequential_41/lstm_101/while/lstm_cell_101/mul_2Mul*sequential_41_lstm_101_while_placeholder_2=sequential_41/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_2
0sequential_41/lstm_101/while/lstm_cell_101/mul_3Mul*sequential_41_lstm_101_while_placeholder_2=sequential_41/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_3ќ
9sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOpReadVariableOpDsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02;
9sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOpб
>sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stackе
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_1е
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_2ў
8sequential_41/lstm_101/while/lstm_cell_101/strided_sliceStridedSliceAsequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp:value:0Gsequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack:output:0Isequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_1:output:0Isequential_41/lstm_101/while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2:
8sequential_41/lstm_101/while/lstm_cell_101/strided_slice
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_4MatMul2sequential_41/lstm_101/while/lstm_cell_101/mul:z:0Asequential_41/lstm_101/while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_4
.sequential_41/lstm_101/while/lstm_cell_101/addAddV2;sequential_41/lstm_101/while/lstm_cell_101/BiasAdd:output:0=sequential_41/lstm_101/while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_41/lstm_101/while/lstm_cell_101/addй
2sequential_41/lstm_101/while/lstm_cell_101/SigmoidSigmoid2sequential_41/lstm_101/while/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_41/lstm_101/while/lstm_cell_101/Sigmoid
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1ReadVariableOpDsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02=
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1е
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2B
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stackй
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_1й
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_2
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_1StridedSliceCsequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1:value:0Isequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_1:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_1Ё
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_5MatMul4sequential_41/lstm_101/while/lstm_cell_101/mul_1:z:0Csequential_41/lstm_101/while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_5
0sequential_41/lstm_101/while/lstm_cell_101/add_1AddV2=sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_1:output:0=sequential_41/lstm_101/while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/add_1п
4sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_1Sigmoid4sequential_41/lstm_101/while/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_1
0sequential_41/lstm_101/while/lstm_cell_101/mul_4Mul8sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_1:y:0*sequential_41_lstm_101_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_4
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2ReadVariableOpDsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02=
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2е
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2B
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stackй
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_1й
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_2
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_2StridedSliceCsequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2:value:0Isequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_1:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_2Ё
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_6MatMul4sequential_41/lstm_101/while/lstm_cell_101/mul_2:z:0Csequential_41/lstm_101/while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_6
0sequential_41/lstm_101/while/lstm_cell_101/add_2AddV2=sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_2:output:0=sequential_41/lstm_101/while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/add_2в
/sequential_41/lstm_101/while/lstm_cell_101/ReluRelu4sequential_41/lstm_101/while/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_41/lstm_101/while/lstm_cell_101/Relu
0sequential_41/lstm_101/while/lstm_cell_101/mul_5Mul6sequential_41/lstm_101/while/lstm_cell_101/Sigmoid:y:0=sequential_41/lstm_101/while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_5
0sequential_41/lstm_101/while/lstm_cell_101/add_3AddV24sequential_41/lstm_101/while/lstm_cell_101/mul_4:z:04sequential_41/lstm_101/while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/add_3
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3ReadVariableOpDsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02=
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3е
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2B
@sequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stackй
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_1й
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2D
Bsequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_2
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_3StridedSliceCsequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3:value:0Isequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_1:output:0Ksequential_41/lstm_101/while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2<
:sequential_41/lstm_101/while/lstm_cell_101/strided_slice_3Ё
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_7MatMul4sequential_41/lstm_101/while/lstm_cell_101/mul_3:z:0Csequential_41/lstm_101/while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 25
3sequential_41/lstm_101/while/lstm_cell_101/MatMul_7
0sequential_41/lstm_101/while/lstm_cell_101/add_4AddV2=sequential_41/lstm_101/while/lstm_cell_101/BiasAdd_3:output:0=sequential_41/lstm_101/while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/add_4п
4sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_2Sigmoid4sequential_41/lstm_101/while/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 26
4sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_2ж
1sequential_41/lstm_101/while/lstm_cell_101/Relu_1Relu4sequential_41/lstm_101/while/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_41/lstm_101/while/lstm_cell_101/Relu_1
0sequential_41/lstm_101/while/lstm_cell_101/mul_6Mul8sequential_41/lstm_101/while/lstm_cell_101/Sigmoid_2:y:0?sequential_41/lstm_101/while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_41/lstm_101/while/lstm_cell_101/mul_6д
Asequential_41/lstm_101/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_41_lstm_101_while_placeholder_1(sequential_41_lstm_101_while_placeholder4sequential_41/lstm_101/while/lstm_cell_101/mul_6:z:0*
_output_shapes
: *
element_dtype02C
Asequential_41/lstm_101/while/TensorArrayV2Write/TensorListSetItem
"sequential_41/lstm_101/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_41/lstm_101/while/add/yХ
 sequential_41/lstm_101/while/addAddV2(sequential_41_lstm_101_while_placeholder+sequential_41/lstm_101/while/add/y:output:0*
T0*
_output_shapes
: 2"
 sequential_41/lstm_101/while/add
$sequential_41/lstm_101/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_41/lstm_101/while/add_1/yщ
"sequential_41/lstm_101/while/add_1AddV2Fsequential_41_lstm_101_while_sequential_41_lstm_101_while_loop_counter-sequential_41/lstm_101/while/add_1/y:output:0*
T0*
_output_shapes
: 2$
"sequential_41/lstm_101/while/add_1Ч
%sequential_41/lstm_101/while/IdentityIdentity&sequential_41/lstm_101/while/add_1:z:0"^sequential_41/lstm_101/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_41/lstm_101/while/Identityё
'sequential_41/lstm_101/while/Identity_1IdentityLsequential_41_lstm_101_while_sequential_41_lstm_101_while_maximum_iterations"^sequential_41/lstm_101/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_41/lstm_101/while/Identity_1Щ
'sequential_41/lstm_101/while/Identity_2Identity$sequential_41/lstm_101/while/add:z:0"^sequential_41/lstm_101/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_41/lstm_101/while/Identity_2і
'sequential_41/lstm_101/while/Identity_3IdentityQsequential_41/lstm_101/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_41/lstm_101/while/NoOp*
T0*
_output_shapes
: 2)
'sequential_41/lstm_101/while/Identity_3ъ
'sequential_41/lstm_101/while/Identity_4Identity4sequential_41/lstm_101/while/lstm_cell_101/mul_6:z:0"^sequential_41/lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_41/lstm_101/while/Identity_4ъ
'sequential_41/lstm_101/while/Identity_5Identity4sequential_41/lstm_101/while/lstm_cell_101/add_3:z:0"^sequential_41/lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_41/lstm_101/while/Identity_5
!sequential_41/lstm_101/while/NoOpNoOp:^sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp<^sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1<^sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2<^sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3@^sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOpB^sequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2#
!sequential_41/lstm_101/while/NoOp"W
%sequential_41_lstm_101_while_identity.sequential_41/lstm_101/while/Identity:output:0"[
'sequential_41_lstm_101_while_identity_10sequential_41/lstm_101/while/Identity_1:output:0"[
'sequential_41_lstm_101_while_identity_20sequential_41/lstm_101/while/Identity_2:output:0"[
'sequential_41_lstm_101_while_identity_30sequential_41/lstm_101/while/Identity_3:output:0"[
'sequential_41_lstm_101_while_identity_40sequential_41/lstm_101/while/Identity_4:output:0"[
'sequential_41_lstm_101_while_identity_50sequential_41/lstm_101/while/Identity_5:output:0"
Bsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resourceDsequential_41_lstm_101_while_lstm_cell_101_readvariableop_resource_0"
Jsequential_41_lstm_101_while_lstm_cell_101_split_1_readvariableop_resourceLsequential_41_lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0"
Hsequential_41_lstm_101_while_lstm_cell_101_split_readvariableop_resourceJsequential_41_lstm_101_while_lstm_cell_101_split_readvariableop_resource_0"
Csequential_41_lstm_101_while_sequential_41_lstm_101_strided_slice_1Esequential_41_lstm_101_while_sequential_41_lstm_101_strided_slice_1_0"
sequential_41_lstm_101_while_tensorarrayv2read_tensorlistgetitem_sequential_41_lstm_101_tensorarrayunstack_tensorlistfromtensorsequential_41_lstm_101_while_tensorarrayv2read_tensorlistgetitem_sequential_41_lstm_101_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2v
9sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp9sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp2z
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_1;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_12z
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_2;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_22z
;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_3;sequential_41/lstm_101/while/lstm_cell_101/ReadVariableOp_32
?sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOp?sequential_41/lstm_101/while/lstm_cell_101/split/ReadVariableOp2
Asequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOpAsequential_41/lstm_101/while/lstm_cell_101/split_1/ReadVariableOp: 
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
/__inference_sequential_41_layer_call_fn_3239401

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
J__inference_sequential_41_layer_call_and_return_conditional_losses_32392202
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
р,
Т
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239290
input_42#
lstm_101_3239259:	
lstm_101_3239261:	#
lstm_101_3239263:	 #
dense_122_3239266:  
dense_122_3239268: #
dense_123_3239271: 
dense_123_3239273:
identityЂ!dense_122/StatefulPartitionedCallЂ!dense_123/StatefulPartitionedCallЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ lstm_101/StatefulPartitionedCallЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp­
 lstm_101/StatefulPartitionedCallStatefulPartitionedCallinput_42lstm_101_3239259lstm_101_3239261lstm_101_3239263*
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32387182"
 lstm_101/StatefulPartitionedCallП
!dense_122/StatefulPartitionedCallStatefulPartitionedCall)lstm_101/StatefulPartitionedCall:output:0dense_122_3239266dense_122_3239268*
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
GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_32387372#
!dense_122/StatefulPartitionedCallР
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_3239271dense_123_3239273*
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
GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_32387592#
!dense_123/StatefulPartitionedCall
reshape_61/PartitionedCallPartitionedCall*dense_123/StatefulPartitionedCall:output:0*
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_32387782
reshape_61/PartitionedCallд
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_101_3239259*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulВ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_3239273*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mul
IdentityIdentity#reshape_61/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall1^dense_123/bias/Regularizer/Square/ReadVariableOp!^lstm_101/StatefulPartitionedCall@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2D
 lstm_101/StatefulPartitionedCall lstm_101/StatefulPartitionedCall2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_42

Б	
while_body_3240167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ј
while/lstm_cell_101/mulMulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mulЌ
while/lstm_cell_101/mul_1Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1Ќ
while/lstm_cell_101/mul_2Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2Ќ
while/lstm_cell_101/mul_3Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
лц
К
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239672

inputsG
4lstm_101_lstm_cell_101_split_readvariableop_resource:	E
6lstm_101_lstm_cell_101_split_1_readvariableop_resource:	A
.lstm_101_lstm_cell_101_readvariableop_resource:	 :
(dense_122_matmul_readvariableop_resource:  7
)dense_122_biasadd_readvariableop_resource: :
(dense_123_matmul_readvariableop_resource: 7
)dense_123_biasadd_readvariableop_resource:
identityЂ dense_122/BiasAdd/ReadVariableOpЂdense_122/MatMul/ReadVariableOpЂ dense_123/BiasAdd/ReadVariableOpЂdense_123/MatMul/ReadVariableOpЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ%lstm_101/lstm_cell_101/ReadVariableOpЂ'lstm_101/lstm_cell_101/ReadVariableOp_1Ђ'lstm_101/lstm_cell_101/ReadVariableOp_2Ђ'lstm_101/lstm_cell_101/ReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂ+lstm_101/lstm_cell_101/split/ReadVariableOpЂ-lstm_101/lstm_cell_101/split_1/ReadVariableOpЂlstm_101/whileV
lstm_101/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_101/Shape
lstm_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_101/strided_slice/stack
lstm_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_101/strided_slice/stack_1
lstm_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_101/strided_slice/stack_2
lstm_101/strided_sliceStridedSlicelstm_101/Shape:output:0%lstm_101/strided_slice/stack:output:0'lstm_101/strided_slice/stack_1:output:0'lstm_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_101/strided_slicen
lstm_101/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros/mul/y
lstm_101/zeros/mulMullstm_101/strided_slice:output:0lstm_101/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros/mulq
lstm_101/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_101/zeros/Less/y
lstm_101/zeros/LessLesslstm_101/zeros/mul:z:0lstm_101/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros/Lesst
lstm_101/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros/packed/1Ї
lstm_101/zeros/packedPacklstm_101/strided_slice:output:0 lstm_101/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_101/zeros/packedq
lstm_101/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/zeros/Const
lstm_101/zerosFilllstm_101/zeros/packed:output:0lstm_101/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/zerosr
lstm_101/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros_1/mul/y
lstm_101/zeros_1/mulMullstm_101/strided_slice:output:0lstm_101/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros_1/mulu
lstm_101/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_101/zeros_1/Less/y
lstm_101/zeros_1/LessLesslstm_101/zeros_1/mul:z:0 lstm_101/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros_1/Lessx
lstm_101/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros_1/packed/1­
lstm_101/zeros_1/packedPacklstm_101/strided_slice:output:0"lstm_101/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_101/zeros_1/packedu
lstm_101/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/zeros_1/ConstЁ
lstm_101/zeros_1Fill lstm_101/zeros_1/packed:output:0lstm_101/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/zeros_1
lstm_101/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_101/transpose/perm
lstm_101/transpose	Transposeinputs lstm_101/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_101/transposej
lstm_101/Shape_1Shapelstm_101/transpose:y:0*
T0*
_output_shapes
:2
lstm_101/Shape_1
lstm_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_101/strided_slice_1/stack
 lstm_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_1/stack_1
 lstm_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_1/stack_2Є
lstm_101/strided_slice_1StridedSlicelstm_101/Shape_1:output:0'lstm_101/strided_slice_1/stack:output:0)lstm_101/strided_slice_1/stack_1:output:0)lstm_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_101/strided_slice_1
$lstm_101/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$lstm_101/TensorArrayV2/element_shapeж
lstm_101/TensorArrayV2TensorListReserve-lstm_101/TensorArrayV2/element_shape:output:0!lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_101/TensorArrayV2б
>lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shape
0lstm_101/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_101/transpose:y:0Glstm_101/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_101/TensorArrayUnstack/TensorListFromTensor
lstm_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_101/strided_slice_2/stack
 lstm_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_2/stack_1
 lstm_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_2/stack_2В
lstm_101/strided_slice_2StridedSlicelstm_101/transpose:y:0'lstm_101/strided_slice_2/stack:output:0)lstm_101/strided_slice_2/stack_1:output:0)lstm_101/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_101/strided_slice_2
&lstm_101/lstm_cell_101/ones_like/ShapeShapelstm_101/zeros:output:0*
T0*
_output_shapes
:2(
&lstm_101/lstm_cell_101/ones_like/Shape
&lstm_101/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm_101/lstm_cell_101/ones_like/Constр
 lstm_101/lstm_cell_101/ones_likeFill/lstm_101/lstm_cell_101/ones_like/Shape:output:0/lstm_101/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/ones_like
&lstm_101/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_101/lstm_cell_101/split/split_dimа
+lstm_101/lstm_cell_101/split/ReadVariableOpReadVariableOp4lstm_101_lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02-
+lstm_101/lstm_cell_101/split/ReadVariableOp
lstm_101/lstm_cell_101/splitSplit/lstm_101/lstm_cell_101/split/split_dim:output:03lstm_101/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_101/lstm_cell_101/splitФ
lstm_101/lstm_cell_101/MatMulMatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/MatMulШ
lstm_101/lstm_cell_101/MatMul_1MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_1Ш
lstm_101/lstm_cell_101/MatMul_2MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_2Ш
lstm_101/lstm_cell_101/MatMul_3MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_3
(lstm_101/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_101/lstm_cell_101/split_1/split_dimв
-lstm_101/lstm_cell_101/split_1/ReadVariableOpReadVariableOp6lstm_101_lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02/
-lstm_101/lstm_cell_101/split_1/ReadVariableOpћ
lstm_101/lstm_cell_101/split_1Split1lstm_101/lstm_cell_101/split_1/split_dim:output:05lstm_101/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2 
lstm_101/lstm_cell_101/split_1Я
lstm_101/lstm_cell_101/BiasAddBiasAdd'lstm_101/lstm_cell_101/MatMul:product:0'lstm_101/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_101/lstm_cell_101/BiasAddе
 lstm_101/lstm_cell_101/BiasAdd_1BiasAdd)lstm_101/lstm_cell_101/MatMul_1:product:0'lstm_101/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_1е
 lstm_101/lstm_cell_101/BiasAdd_2BiasAdd)lstm_101/lstm_cell_101/MatMul_2:product:0'lstm_101/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_2е
 lstm_101/lstm_cell_101/BiasAdd_3BiasAdd)lstm_101/lstm_cell_101/MatMul_3:product:0'lstm_101/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_3Е
lstm_101/lstm_cell_101/mulMullstm_101/zeros:output:0)lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mulЙ
lstm_101/lstm_cell_101/mul_1Mullstm_101/zeros:output:0)lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_1Й
lstm_101/lstm_cell_101/mul_2Mullstm_101/zeros:output:0)lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_2Й
lstm_101/lstm_cell_101/mul_3Mullstm_101/zeros:output:0)lstm_101/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_3О
%lstm_101/lstm_cell_101/ReadVariableOpReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_101/lstm_cell_101/ReadVariableOpЉ
*lstm_101/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_101/lstm_cell_101/strided_slice/stack­
,lstm_101/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_101/lstm_cell_101/strided_slice/stack_1­
,lstm_101/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_101/lstm_cell_101/strided_slice/stack_2
$lstm_101/lstm_cell_101/strided_sliceStridedSlice-lstm_101/lstm_cell_101/ReadVariableOp:value:03lstm_101/lstm_cell_101/strided_slice/stack:output:05lstm_101/lstm_cell_101/strided_slice/stack_1:output:05lstm_101/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_101/lstm_cell_101/strided_sliceЭ
lstm_101/lstm_cell_101/MatMul_4MatMullstm_101/lstm_cell_101/mul:z:0-lstm_101/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_4Ч
lstm_101/lstm_cell_101/addAddV2'lstm_101/lstm_cell_101/BiasAdd:output:0)lstm_101/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add
lstm_101/lstm_cell_101/SigmoidSigmoidlstm_101/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_101/lstm_cell_101/SigmoidТ
'lstm_101/lstm_cell_101/ReadVariableOp_1ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_1­
,lstm_101/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_101/lstm_cell_101/strided_slice_1/stackБ
.lstm_101/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_101/lstm_cell_101/strided_slice_1/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_1/stack_2
&lstm_101/lstm_cell_101/strided_slice_1StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_1:value:05lstm_101/lstm_cell_101/strided_slice_1/stack:output:07lstm_101/lstm_cell_101/strided_slice_1/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_1б
lstm_101/lstm_cell_101/MatMul_5MatMul lstm_101/lstm_cell_101/mul_1:z:0/lstm_101/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_5Э
lstm_101/lstm_cell_101/add_1AddV2)lstm_101/lstm_cell_101/BiasAdd_1:output:0)lstm_101/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_1Ѓ
 lstm_101/lstm_cell_101/Sigmoid_1Sigmoid lstm_101/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/Sigmoid_1Ж
lstm_101/lstm_cell_101/mul_4Mul$lstm_101/lstm_cell_101/Sigmoid_1:y:0lstm_101/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_4Т
'lstm_101/lstm_cell_101/ReadVariableOp_2ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_2­
,lstm_101/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_101/lstm_cell_101/strided_slice_2/stackБ
.lstm_101/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_101/lstm_cell_101/strided_slice_2/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_2/stack_2
&lstm_101/lstm_cell_101/strided_slice_2StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_2:value:05lstm_101/lstm_cell_101/strided_slice_2/stack:output:07lstm_101/lstm_cell_101/strided_slice_2/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_2б
lstm_101/lstm_cell_101/MatMul_6MatMul lstm_101/lstm_cell_101/mul_2:z:0/lstm_101/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_6Э
lstm_101/lstm_cell_101/add_2AddV2)lstm_101/lstm_cell_101/BiasAdd_2:output:0)lstm_101/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_2
lstm_101/lstm_cell_101/ReluRelu lstm_101/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/ReluФ
lstm_101/lstm_cell_101/mul_5Mul"lstm_101/lstm_cell_101/Sigmoid:y:0)lstm_101/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_5Л
lstm_101/lstm_cell_101/add_3AddV2 lstm_101/lstm_cell_101/mul_4:z:0 lstm_101/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_3Т
'lstm_101/lstm_cell_101/ReadVariableOp_3ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_3­
,lstm_101/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_101/lstm_cell_101/strided_slice_3/stackБ
.lstm_101/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_101/lstm_cell_101/strided_slice_3/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_3/stack_2
&lstm_101/lstm_cell_101/strided_slice_3StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_3:value:05lstm_101/lstm_cell_101/strided_slice_3/stack:output:07lstm_101/lstm_cell_101/strided_slice_3/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_3б
lstm_101/lstm_cell_101/MatMul_7MatMul lstm_101/lstm_cell_101/mul_3:z:0/lstm_101/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_7Э
lstm_101/lstm_cell_101/add_4AddV2)lstm_101/lstm_cell_101/BiasAdd_3:output:0)lstm_101/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_4Ѓ
 lstm_101/lstm_cell_101/Sigmoid_2Sigmoid lstm_101/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/Sigmoid_2
lstm_101/lstm_cell_101/Relu_1Relu lstm_101/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/Relu_1Ш
lstm_101/lstm_cell_101/mul_6Mul$lstm_101/lstm_cell_101/Sigmoid_2:y:0+lstm_101/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_6Ё
&lstm_101/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&lstm_101/TensorArrayV2_1/element_shapeм
lstm_101/TensorArrayV2_1TensorListReserve/lstm_101/TensorArrayV2_1/element_shape:output:0!lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_101/TensorArrayV2_1`
lstm_101/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/time
!lstm_101/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!lstm_101/while/maximum_iterations|
lstm_101/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/while/loop_counter
lstm_101/whileWhile$lstm_101/while/loop_counter:output:0*lstm_101/while/maximum_iterations:output:0lstm_101/time:output:0!lstm_101/TensorArrayV2_1:handle:0lstm_101/zeros:output:0lstm_101/zeros_1:output:0!lstm_101/strided_slice_1:output:0@lstm_101/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_101_lstm_cell_101_split_readvariableop_resource6lstm_101_lstm_cell_101_split_1_readvariableop_resource.lstm_101_lstm_cell_101_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_101_while_body_3239511*'
condR
lstm_101_while_cond_3239510*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_101/whileЧ
9lstm_101/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2;
9lstm_101/TensorArrayV2Stack/TensorListStack/element_shape
+lstm_101/TensorArrayV2Stack/TensorListStackTensorListStacklstm_101/while:output:3Blstm_101/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02-
+lstm_101/TensorArrayV2Stack/TensorListStack
lstm_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
lstm_101/strided_slice_3/stack
 lstm_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_101/strided_slice_3/stack_1
 lstm_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_3/stack_2а
lstm_101/strided_slice_3StridedSlice4lstm_101/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_101/strided_slice_3/stack:output:0)lstm_101/strided_slice_3/stack_1:output:0)lstm_101/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_101/strided_slice_3
lstm_101/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_101/transpose_1/permЩ
lstm_101/transpose_1	Transpose4lstm_101/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_101/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_101/transpose_1x
lstm_101/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/runtimeЋ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense_122/MatMul/ReadVariableOpЌ
dense_122/MatMulMatMul!lstm_101/strided_slice_3:output:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/MatMulЊ
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_122/BiasAdd/ReadVariableOpЉ
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/BiasAddv
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/ReluЋ
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense_123/MatMul/ReadVariableOpЇ
dense_123/MatMulMatMuldense_122/Relu:activations:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_123/MatMulЊ
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_123/BiasAdd/ReadVariableOpЉ
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_123/BiasAddn
reshape_61/ShapeShapedense_123/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_61/Shape
reshape_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_61/strided_slice/stack
 reshape_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_61/strided_slice/stack_1
 reshape_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_61/strided_slice/stack_2Є
reshape_61/strided_sliceStridedSlicereshape_61/Shape:output:0'reshape_61/strided_slice/stack:output:0)reshape_61/strided_slice/stack_1:output:0)reshape_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_61/strided_slicez
reshape_61/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_61/Reshape/shape/1z
reshape_61/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_61/Reshape/shape/2з
reshape_61/Reshape/shapePack!reshape_61/strided_slice:output:0#reshape_61/Reshape/shape/1:output:0#reshape_61/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_61/Reshape/shapeЈ
reshape_61/ReshapeReshapedense_123/BiasAdd:output:0!reshape_61/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_61/Reshapeј
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_101_lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulЪ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mulz
IdentityIdentityreshape_61/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityт
NoOpNoOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp1^dense_123/bias/Regularizer/Square/ReadVariableOp&^lstm_101/lstm_cell_101/ReadVariableOp(^lstm_101/lstm_cell_101/ReadVariableOp_1(^lstm_101/lstm_cell_101/ReadVariableOp_2(^lstm_101/lstm_cell_101/ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp,^lstm_101/lstm_cell_101/split/ReadVariableOp.^lstm_101/lstm_cell_101/split_1/ReadVariableOp^lstm_101/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2N
%lstm_101/lstm_cell_101/ReadVariableOp%lstm_101/lstm_cell_101/ReadVariableOp2R
'lstm_101/lstm_cell_101/ReadVariableOp_1'lstm_101/lstm_cell_101/ReadVariableOp_12R
'lstm_101/lstm_cell_101/ReadVariableOp_2'lstm_101/lstm_cell_101/ReadVariableOp_22R
'lstm_101/lstm_cell_101/ReadVariableOp_3'lstm_101/lstm_cell_101/ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2Z
+lstm_101/lstm_cell_101/split/ReadVariableOp+lstm_101/lstm_cell_101/split/ReadVariableOp2^
-lstm_101/lstm_cell_101/split_1/ReadVariableOp-lstm_101/lstm_cell_101/split_1/ReadVariableOp2 
lstm_101/whilelstm_101/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


)sequential_41_lstm_101_while_cond_3237535J
Fsequential_41_lstm_101_while_sequential_41_lstm_101_while_loop_counterP
Lsequential_41_lstm_101_while_sequential_41_lstm_101_while_maximum_iterations,
(sequential_41_lstm_101_while_placeholder.
*sequential_41_lstm_101_while_placeholder_1.
*sequential_41_lstm_101_while_placeholder_2.
*sequential_41_lstm_101_while_placeholder_3L
Hsequential_41_lstm_101_while_less_sequential_41_lstm_101_strided_slice_1c
_sequential_41_lstm_101_while_sequential_41_lstm_101_while_cond_3237535___redundant_placeholder0c
_sequential_41_lstm_101_while_sequential_41_lstm_101_while_cond_3237535___redundant_placeholder1c
_sequential_41_lstm_101_while_sequential_41_lstm_101_while_cond_3237535___redundant_placeholder2c
_sequential_41_lstm_101_while_sequential_41_lstm_101_while_cond_3237535___redundant_placeholder3)
%sequential_41_lstm_101_while_identity
у
!sequential_41/lstm_101/while/LessLess(sequential_41_lstm_101_while_placeholderHsequential_41_lstm_101_while_less_sequential_41_lstm_101_strided_slice_1*
T0*
_output_shapes
: 2#
!sequential_41/lstm_101/while/LessЂ
%sequential_41/lstm_101/while/IdentityIdentity%sequential_41/lstm_101/while/Less:z:0*
T0
*
_output_shapes
: 2'
%sequential_41/lstm_101/while/Identity"W
%sequential_41_lstm_101_while_identity.sequential_41/lstm_101/while/Identity:output:0*(
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
while_cond_3240166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3240166___redundant_placeholder05
1while_while_cond_3240166___redundant_placeholder15
1while_while_cond_3240166___redundant_placeholder25
1while_while_cond_3240166___redundant_placeholder3
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
&
ё
while_body_3237823
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_101_3237847_0:	,
while_lstm_cell_101_3237849_0:	0
while_lstm_cell_101_3237851_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_101_3237847:	*
while_lstm_cell_101_3237849:	.
while_lstm_cell_101_3237851:	 Ђ+while/lstm_cell_101/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_101/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_101_3237847_0while_lstm_cell_101_3237849_0while_lstm_cell_101_3237851_0*
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32378092-
+while/lstm_cell_101/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_101/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_101/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_101/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_101/StatefulPartitionedCall*"
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
while_lstm_cell_101_3237847while_lstm_cell_101_3237847_0"<
while_lstm_cell_101_3237849while_lstm_cell_101_3237849_0"<
while_lstm_cell_101_3237851while_lstm_cell_101_3237851_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2Z
+while/lstm_cell_101/StatefulPartitionedCall+while/lstm_cell_101/StatefulPartitionedCall: 
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

c
G__inference_reshape_61_layer_call_and_return_conditional_losses_3241226

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
Њ
ь
lstm_101_while_body_3239511.
*lstm_101_while_lstm_101_while_loop_counter4
0lstm_101_while_lstm_101_while_maximum_iterations
lstm_101_while_placeholder 
lstm_101_while_placeholder_1 
lstm_101_while_placeholder_2 
lstm_101_while_placeholder_3-
)lstm_101_while_lstm_101_strided_slice_1_0i
elstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0:	M
>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0:	I
6lstm_101_while_lstm_cell_101_readvariableop_resource_0:	 
lstm_101_while_identity
lstm_101_while_identity_1
lstm_101_while_identity_2
lstm_101_while_identity_3
lstm_101_while_identity_4
lstm_101_while_identity_5+
'lstm_101_while_lstm_101_strided_slice_1g
clstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensorM
:lstm_101_while_lstm_cell_101_split_readvariableop_resource:	K
<lstm_101_while_lstm_cell_101_split_1_readvariableop_resource:	G
4lstm_101_while_lstm_cell_101_readvariableop_resource:	 Ђ+lstm_101/while/lstm_cell_101/ReadVariableOpЂ-lstm_101/while/lstm_cell_101/ReadVariableOp_1Ђ-lstm_101/while/lstm_cell_101/ReadVariableOp_2Ђ-lstm_101/while/lstm_cell_101/ReadVariableOp_3Ђ1lstm_101/while/lstm_cell_101/split/ReadVariableOpЂ3lstm_101/while/lstm_cell_101/split_1/ReadVariableOpе
@lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2B
@lstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shape
2lstm_101/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0lstm_101_while_placeholderIlstm_101/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype024
2lstm_101/while/TensorArrayV2Read/TensorListGetItemЈ
,lstm_101/while/lstm_cell_101/ones_like/ShapeShapelstm_101_while_placeholder_2*
T0*
_output_shapes
:2.
,lstm_101/while/lstm_cell_101/ones_like/ShapeЁ
,lstm_101/while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,lstm_101/while/lstm_cell_101/ones_like/Constј
&lstm_101/while/lstm_cell_101/ones_likeFill5lstm_101/while/lstm_cell_101/ones_like/Shape:output:05lstm_101/while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/ones_like
,lstm_101/while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_101/while/lstm_cell_101/split/split_dimф
1lstm_101/while/lstm_cell_101/split/ReadVariableOpReadVariableOp<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1lstm_101/while/lstm_cell_101/split/ReadVariableOp
"lstm_101/while/lstm_cell_101/splitSplit5lstm_101/while/lstm_cell_101/split/split_dim:output:09lstm_101/while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2$
"lstm_101/while/lstm_cell_101/splitю
#lstm_101/while/lstm_cell_101/MatMulMatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_101/while/lstm_cell_101/MatMulђ
%lstm_101/while/lstm_cell_101/MatMul_1MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_1ђ
%lstm_101/while/lstm_cell_101/MatMul_2MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_2ђ
%lstm_101/while/lstm_cell_101/MatMul_3MatMul9lstm_101/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_101/while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_3Ђ
.lstm_101/while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.lstm_101/while/lstm_cell_101/split_1/split_dimц
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype025
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp
$lstm_101/while/lstm_cell_101/split_1Split7lstm_101/while/lstm_cell_101/split_1/split_dim:output:0;lstm_101/while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2&
$lstm_101/while/lstm_cell_101/split_1ч
$lstm_101/while/lstm_cell_101/BiasAddBiasAdd-lstm_101/while/lstm_cell_101/MatMul:product:0-lstm_101/while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/while/lstm_cell_101/BiasAddэ
&lstm_101/while/lstm_cell_101/BiasAdd_1BiasAdd/lstm_101/while/lstm_cell_101/MatMul_1:product:0-lstm_101/while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_1э
&lstm_101/while/lstm_cell_101/BiasAdd_2BiasAdd/lstm_101/while/lstm_cell_101/MatMul_2:product:0-lstm_101/while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_2э
&lstm_101/while/lstm_cell_101/BiasAdd_3BiasAdd/lstm_101/while/lstm_cell_101/MatMul_3:product:0-lstm_101/while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/BiasAdd_3Ь
 lstm_101/while/lstm_cell_101/mulMullstm_101_while_placeholder_2/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/while/lstm_cell_101/mulа
"lstm_101/while/lstm_cell_101/mul_1Mullstm_101_while_placeholder_2/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_1а
"lstm_101/while/lstm_cell_101/mul_2Mullstm_101_while_placeholder_2/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_2а
"lstm_101/while/lstm_cell_101/mul_3Mullstm_101_while_placeholder_2/lstm_101/while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_3в
+lstm_101/while/lstm_cell_101/ReadVariableOpReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_101/while/lstm_cell_101/ReadVariableOpЕ
0lstm_101/while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_101/while/lstm_cell_101/strided_slice/stackЙ
2lstm_101/while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_101/while/lstm_cell_101/strided_slice/stack_1Й
2lstm_101/while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_101/while/lstm_cell_101/strided_slice/stack_2Њ
*lstm_101/while/lstm_cell_101/strided_sliceStridedSlice3lstm_101/while/lstm_cell_101/ReadVariableOp:value:09lstm_101/while/lstm_cell_101/strided_slice/stack:output:0;lstm_101/while/lstm_cell_101/strided_slice/stack_1:output:0;lstm_101/while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_101/while/lstm_cell_101/strided_sliceх
%lstm_101/while/lstm_cell_101/MatMul_4MatMul$lstm_101/while/lstm_cell_101/mul:z:03lstm_101/while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_4п
 lstm_101/while/lstm_cell_101/addAddV2-lstm_101/while/lstm_cell_101/BiasAdd:output:0/lstm_101/while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/while/lstm_cell_101/addЏ
$lstm_101/while/lstm_cell_101/SigmoidSigmoid$lstm_101/while/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/while/lstm_cell_101/Sigmoidж
-lstm_101/while/lstm_cell_101/ReadVariableOp_1ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_1Й
2lstm_101/while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_101/while/lstm_cell_101/strided_slice_1/stackН
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   26
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_1/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_1StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_1:value:0;lstm_101/while/lstm_cell_101/strided_slice_1/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_1/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_1щ
%lstm_101/while/lstm_cell_101/MatMul_5MatMul&lstm_101/while/lstm_cell_101/mul_1:z:05lstm_101/while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_5х
"lstm_101/while/lstm_cell_101/add_1AddV2/lstm_101/while/lstm_cell_101/BiasAdd_1:output:0/lstm_101/while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_1Е
&lstm_101/while/lstm_cell_101/Sigmoid_1Sigmoid&lstm_101/while/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/Sigmoid_1Ы
"lstm_101/while/lstm_cell_101/mul_4Mul*lstm_101/while/lstm_cell_101/Sigmoid_1:y:0lstm_101_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_4ж
-lstm_101/while/lstm_cell_101/ReadVariableOp_2ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_2Й
2lstm_101/while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_101/while/lstm_cell_101/strided_slice_2/stackН
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   26
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_2/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_2StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_2:value:0;lstm_101/while/lstm_cell_101/strided_slice_2/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_2/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_2щ
%lstm_101/while/lstm_cell_101/MatMul_6MatMul&lstm_101/while/lstm_cell_101/mul_2:z:05lstm_101/while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_6х
"lstm_101/while/lstm_cell_101/add_2AddV2/lstm_101/while/lstm_cell_101/BiasAdd_2:output:0/lstm_101/while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_2Ј
!lstm_101/while/lstm_cell_101/ReluRelu&lstm_101/while/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_101/while/lstm_cell_101/Reluм
"lstm_101/while/lstm_cell_101/mul_5Mul(lstm_101/while/lstm_cell_101/Sigmoid:y:0/lstm_101/while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_5г
"lstm_101/while/lstm_cell_101/add_3AddV2&lstm_101/while/lstm_cell_101/mul_4:z:0&lstm_101/while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_3ж
-lstm_101/while/lstm_cell_101/ReadVariableOp_3ReadVariableOp6lstm_101_while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02/
-lstm_101/while/lstm_cell_101/ReadVariableOp_3Й
2lstm_101/while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_101/while/lstm_cell_101/strided_slice_3/stackН
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_1Н
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4lstm_101/while/lstm_cell_101/strided_slice_3/stack_2Ж
,lstm_101/while/lstm_cell_101/strided_slice_3StridedSlice5lstm_101/while/lstm_cell_101/ReadVariableOp_3:value:0;lstm_101/while/lstm_cell_101/strided_slice_3/stack:output:0=lstm_101/while/lstm_cell_101/strided_slice_3/stack_1:output:0=lstm_101/while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,lstm_101/while/lstm_cell_101/strided_slice_3щ
%lstm_101/while/lstm_cell_101/MatMul_7MatMul&lstm_101/while/lstm_cell_101/mul_3:z:05lstm_101/while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/while/lstm_cell_101/MatMul_7х
"lstm_101/while/lstm_cell_101/add_4AddV2/lstm_101/while/lstm_cell_101/BiasAdd_3:output:0/lstm_101/while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/add_4Е
&lstm_101/while/lstm_cell_101/Sigmoid_2Sigmoid&lstm_101/while/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/while/lstm_cell_101/Sigmoid_2Ќ
#lstm_101/while/lstm_cell_101/Relu_1Relu&lstm_101/while/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_101/while/lstm_cell_101/Relu_1р
"lstm_101/while/lstm_cell_101/mul_6Mul*lstm_101/while/lstm_cell_101/Sigmoid_2:y:01lstm_101/while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/while/lstm_cell_101/mul_6
3lstm_101/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_101_while_placeholder_1lstm_101_while_placeholder&lstm_101/while/lstm_cell_101/mul_6:z:0*
_output_shapes
: *
element_dtype025
3lstm_101/while/TensorArrayV2Write/TensorListSetItemn
lstm_101/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_101/while/add/y
lstm_101/while/addAddV2lstm_101_while_placeholderlstm_101/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_101/while/addr
lstm_101/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_101/while/add_1/yЃ
lstm_101/while/add_1AddV2*lstm_101_while_lstm_101_while_loop_counterlstm_101/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_101/while/add_1
lstm_101/while/IdentityIdentitylstm_101/while/add_1:z:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/IdentityЋ
lstm_101/while/Identity_1Identity0lstm_101_while_lstm_101_while_maximum_iterations^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_1
lstm_101/while/Identity_2Identitylstm_101/while/add:z:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_2О
lstm_101/while/Identity_3IdentityClstm_101/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_101/while/NoOp*
T0*
_output_shapes
: 2
lstm_101/while/Identity_3В
lstm_101/while/Identity_4Identity&lstm_101/while/lstm_cell_101/mul_6:z:0^lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/while/Identity_4В
lstm_101/while/Identity_5Identity&lstm_101/while/lstm_cell_101/add_3:z:0^lstm_101/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/while/Identity_5
lstm_101/while/NoOpNoOp,^lstm_101/while/lstm_cell_101/ReadVariableOp.^lstm_101/while/lstm_cell_101/ReadVariableOp_1.^lstm_101/while/lstm_cell_101/ReadVariableOp_2.^lstm_101/while/lstm_cell_101/ReadVariableOp_32^lstm_101/while/lstm_cell_101/split/ReadVariableOp4^lstm_101/while/lstm_cell_101/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_101/while/NoOp";
lstm_101_while_identity lstm_101/while/Identity:output:0"?
lstm_101_while_identity_1"lstm_101/while/Identity_1:output:0"?
lstm_101_while_identity_2"lstm_101/while/Identity_2:output:0"?
lstm_101_while_identity_3"lstm_101/while/Identity_3:output:0"?
lstm_101_while_identity_4"lstm_101/while/Identity_4:output:0"?
lstm_101_while_identity_5"lstm_101/while/Identity_5:output:0"T
'lstm_101_while_lstm_101_strided_slice_1)lstm_101_while_lstm_101_strided_slice_1_0"n
4lstm_101_while_lstm_cell_101_readvariableop_resource6lstm_101_while_lstm_cell_101_readvariableop_resource_0"~
<lstm_101_while_lstm_cell_101_split_1_readvariableop_resource>lstm_101_while_lstm_cell_101_split_1_readvariableop_resource_0"z
:lstm_101_while_lstm_cell_101_split_readvariableop_resource<lstm_101_while_lstm_cell_101_split_readvariableop_resource_0"Ь
clstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensorelstm_101_while_tensorarrayv2read_tensorlistgetitem_lstm_101_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2Z
+lstm_101/while/lstm_cell_101/ReadVariableOp+lstm_101/while/lstm_cell_101/ReadVariableOp2^
-lstm_101/while/lstm_cell_101/ReadVariableOp_1-lstm_101/while/lstm_cell_101/ReadVariableOp_12^
-lstm_101/while/lstm_cell_101/ReadVariableOp_2-lstm_101/while/lstm_cell_101/ReadVariableOp_22^
-lstm_101/while/lstm_cell_101/ReadVariableOp_3-lstm_101/while/lstm_cell_101/ReadVariableOp_32f
1lstm_101/while/lstm_cell_101/split/ReadVariableOp1lstm_101/while/lstm_cell_101/split/ReadVariableOp2j
3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp3lstm_101/while/lstm_cell_101/split_1/ReadVariableOp: 
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
О
К
J__inference_sequential_41_layer_call_and_return_conditional_losses_3240007

inputsG
4lstm_101_lstm_cell_101_split_readvariableop_resource:	E
6lstm_101_lstm_cell_101_split_1_readvariableop_resource:	A
.lstm_101_lstm_cell_101_readvariableop_resource:	 :
(dense_122_matmul_readvariableop_resource:  7
)dense_122_biasadd_readvariableop_resource: :
(dense_123_matmul_readvariableop_resource: 7
)dense_123_biasadd_readvariableop_resource:
identityЂ dense_122/BiasAdd/ReadVariableOpЂdense_122/MatMul/ReadVariableOpЂ dense_123/BiasAdd/ReadVariableOpЂdense_123/MatMul/ReadVariableOpЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ%lstm_101/lstm_cell_101/ReadVariableOpЂ'lstm_101/lstm_cell_101/ReadVariableOp_1Ђ'lstm_101/lstm_cell_101/ReadVariableOp_2Ђ'lstm_101/lstm_cell_101/ReadVariableOp_3Ђ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂ+lstm_101/lstm_cell_101/split/ReadVariableOpЂ-lstm_101/lstm_cell_101/split_1/ReadVariableOpЂlstm_101/whileV
lstm_101/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_101/Shape
lstm_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_101/strided_slice/stack
lstm_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_101/strided_slice/stack_1
lstm_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_101/strided_slice/stack_2
lstm_101/strided_sliceStridedSlicelstm_101/Shape:output:0%lstm_101/strided_slice/stack:output:0'lstm_101/strided_slice/stack_1:output:0'lstm_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_101/strided_slicen
lstm_101/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros/mul/y
lstm_101/zeros/mulMullstm_101/strided_slice:output:0lstm_101/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros/mulq
lstm_101/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_101/zeros/Less/y
lstm_101/zeros/LessLesslstm_101/zeros/mul:z:0lstm_101/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros/Lesst
lstm_101/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros/packed/1Ї
lstm_101/zeros/packedPacklstm_101/strided_slice:output:0 lstm_101/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_101/zeros/packedq
lstm_101/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/zeros/Const
lstm_101/zerosFilllstm_101/zeros/packed:output:0lstm_101/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/zerosr
lstm_101/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros_1/mul/y
lstm_101/zeros_1/mulMullstm_101/strided_slice:output:0lstm_101/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros_1/mulu
lstm_101/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_101/zeros_1/Less/y
lstm_101/zeros_1/LessLesslstm_101/zeros_1/mul:z:0 lstm_101/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_101/zeros_1/Lessx
lstm_101/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/zeros_1/packed/1­
lstm_101/zeros_1/packedPacklstm_101/strided_slice:output:0"lstm_101/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_101/zeros_1/packedu
lstm_101/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/zeros_1/ConstЁ
lstm_101/zeros_1Fill lstm_101/zeros_1/packed:output:0lstm_101/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/zeros_1
lstm_101/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_101/transpose/perm
lstm_101/transpose	Transposeinputs lstm_101/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_101/transposej
lstm_101/Shape_1Shapelstm_101/transpose:y:0*
T0*
_output_shapes
:2
lstm_101/Shape_1
lstm_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_101/strided_slice_1/stack
 lstm_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_1/stack_1
 lstm_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_1/stack_2Є
lstm_101/strided_slice_1StridedSlicelstm_101/Shape_1:output:0'lstm_101/strided_slice_1/stack:output:0)lstm_101/strided_slice_1/stack_1:output:0)lstm_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_101/strided_slice_1
$lstm_101/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$lstm_101/TensorArrayV2/element_shapeж
lstm_101/TensorArrayV2TensorListReserve-lstm_101/TensorArrayV2/element_shape:output:0!lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_101/TensorArrayV2б
>lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>lstm_101/TensorArrayUnstack/TensorListFromTensor/element_shape
0lstm_101/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_101/transpose:y:0Glstm_101/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_101/TensorArrayUnstack/TensorListFromTensor
lstm_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_101/strided_slice_2/stack
 lstm_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_2/stack_1
 lstm_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_2/stack_2В
lstm_101/strided_slice_2StridedSlicelstm_101/transpose:y:0'lstm_101/strided_slice_2/stack:output:0)lstm_101/strided_slice_2/stack_1:output:0)lstm_101/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_101/strided_slice_2
&lstm_101/lstm_cell_101/ones_like/ShapeShapelstm_101/zeros:output:0*
T0*
_output_shapes
:2(
&lstm_101/lstm_cell_101/ones_like/Shape
&lstm_101/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm_101/lstm_cell_101/ones_like/Constр
 lstm_101/lstm_cell_101/ones_likeFill/lstm_101/lstm_cell_101/ones_like/Shape:output:0/lstm_101/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/ones_like
$lstm_101/lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_101/lstm_cell_101/dropout/Constл
"lstm_101/lstm_cell_101/dropout/MulMul)lstm_101/lstm_cell_101/ones_like:output:0-lstm_101/lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_101/lstm_cell_101/dropout/MulЅ
$lstm_101/lstm_cell_101/dropout/ShapeShape)lstm_101/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_101/lstm_cell_101/dropout/Shape
;lstm_101/lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform-lstm_101/lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2НЅ~2=
;lstm_101/lstm_cell_101/dropout/random_uniform/RandomUniformЃ
-lstm_101/lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_101/lstm_cell_101/dropout/GreaterEqual/y
+lstm_101/lstm_cell_101/dropout/GreaterEqualGreaterEqualDlstm_101/lstm_cell_101/dropout/random_uniform/RandomUniform:output:06lstm_101/lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_101/lstm_cell_101/dropout/GreaterEqualФ
#lstm_101/lstm_cell_101/dropout/CastCast/lstm_101/lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_101/lstm_cell_101/dropout/Castж
$lstm_101/lstm_cell_101/dropout/Mul_1Mul&lstm_101/lstm_cell_101/dropout/Mul:z:0'lstm_101/lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/lstm_cell_101/dropout/Mul_1
&lstm_101/lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2(
&lstm_101/lstm_cell_101/dropout_1/Constс
$lstm_101/lstm_cell_101/dropout_1/MulMul)lstm_101/lstm_cell_101/ones_like:output:0/lstm_101/lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/lstm_cell_101/dropout_1/MulЉ
&lstm_101/lstm_cell_101/dropout_1/ShapeShape)lstm_101/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_101/lstm_cell_101/dropout_1/Shape
=lstm_101/lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform/lstm_101/lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2 Пљ2?
=lstm_101/lstm_cell_101/dropout_1/random_uniform/RandomUniformЇ
/lstm_101/lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>21
/lstm_101/lstm_cell_101/dropout_1/GreaterEqual/yЂ
-lstm_101/lstm_cell_101/dropout_1/GreaterEqualGreaterEqualFlstm_101/lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:08lstm_101/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-lstm_101/lstm_cell_101/dropout_1/GreaterEqualЪ
%lstm_101/lstm_cell_101/dropout_1/CastCast1lstm_101/lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/lstm_cell_101/dropout_1/Castо
&lstm_101/lstm_cell_101/dropout_1/Mul_1Mul(lstm_101/lstm_cell_101/dropout_1/Mul:z:0)lstm_101/lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/lstm_cell_101/dropout_1/Mul_1
&lstm_101/lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2(
&lstm_101/lstm_cell_101/dropout_2/Constс
$lstm_101/lstm_cell_101/dropout_2/MulMul)lstm_101/lstm_cell_101/ones_like:output:0/lstm_101/lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/lstm_cell_101/dropout_2/MulЉ
&lstm_101/lstm_cell_101/dropout_2/ShapeShape)lstm_101/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_101/lstm_cell_101/dropout_2/Shape
=lstm_101/lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform/lstm_101/lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2жУ2?
=lstm_101/lstm_cell_101/dropout_2/random_uniform/RandomUniformЇ
/lstm_101/lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>21
/lstm_101/lstm_cell_101/dropout_2/GreaterEqual/yЂ
-lstm_101/lstm_cell_101/dropout_2/GreaterEqualGreaterEqualFlstm_101/lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:08lstm_101/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-lstm_101/lstm_cell_101/dropout_2/GreaterEqualЪ
%lstm_101/lstm_cell_101/dropout_2/CastCast1lstm_101/lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/lstm_cell_101/dropout_2/Castо
&lstm_101/lstm_cell_101/dropout_2/Mul_1Mul(lstm_101/lstm_cell_101/dropout_2/Mul:z:0)lstm_101/lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/lstm_cell_101/dropout_2/Mul_1
&lstm_101/lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2(
&lstm_101/lstm_cell_101/dropout_3/Constс
$lstm_101/lstm_cell_101/dropout_3/MulMul)lstm_101/lstm_cell_101/ones_like:output:0/lstm_101/lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_101/lstm_cell_101/dropout_3/MulЉ
&lstm_101/lstm_cell_101/dropout_3/ShapeShape)lstm_101/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_101/lstm_cell_101/dropout_3/Shape
=lstm_101/lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform/lstm_101/lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ХЯА2?
=lstm_101/lstm_cell_101/dropout_3/random_uniform/RandomUniformЇ
/lstm_101/lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>21
/lstm_101/lstm_cell_101/dropout_3/GreaterEqual/yЂ
-lstm_101/lstm_cell_101/dropout_3/GreaterEqualGreaterEqualFlstm_101/lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:08lstm_101/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-lstm_101/lstm_cell_101/dropout_3/GreaterEqualЪ
%lstm_101/lstm_cell_101/dropout_3/CastCast1lstm_101/lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_101/lstm_cell_101/dropout_3/Castо
&lstm_101/lstm_cell_101/dropout_3/Mul_1Mul(lstm_101/lstm_cell_101/dropout_3/Mul:z:0)lstm_101/lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_101/lstm_cell_101/dropout_3/Mul_1
&lstm_101/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_101/lstm_cell_101/split/split_dimа
+lstm_101/lstm_cell_101/split/ReadVariableOpReadVariableOp4lstm_101_lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02-
+lstm_101/lstm_cell_101/split/ReadVariableOp
lstm_101/lstm_cell_101/splitSplit/lstm_101/lstm_cell_101/split/split_dim:output:03lstm_101/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_101/lstm_cell_101/splitФ
lstm_101/lstm_cell_101/MatMulMatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/MatMulШ
lstm_101/lstm_cell_101/MatMul_1MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_1Ш
lstm_101/lstm_cell_101/MatMul_2MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_2Ш
lstm_101/lstm_cell_101/MatMul_3MatMul!lstm_101/strided_slice_2:output:0%lstm_101/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_3
(lstm_101/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_101/lstm_cell_101/split_1/split_dimв
-lstm_101/lstm_cell_101/split_1/ReadVariableOpReadVariableOp6lstm_101_lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02/
-lstm_101/lstm_cell_101/split_1/ReadVariableOpћ
lstm_101/lstm_cell_101/split_1Split1lstm_101/lstm_cell_101/split_1/split_dim:output:05lstm_101/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2 
lstm_101/lstm_cell_101/split_1Я
lstm_101/lstm_cell_101/BiasAddBiasAdd'lstm_101/lstm_cell_101/MatMul:product:0'lstm_101/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_101/lstm_cell_101/BiasAddе
 lstm_101/lstm_cell_101/BiasAdd_1BiasAdd)lstm_101/lstm_cell_101/MatMul_1:product:0'lstm_101/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_1е
 lstm_101/lstm_cell_101/BiasAdd_2BiasAdd)lstm_101/lstm_cell_101/MatMul_2:product:0'lstm_101/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_2е
 lstm_101/lstm_cell_101/BiasAdd_3BiasAdd)lstm_101/lstm_cell_101/MatMul_3:product:0'lstm_101/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/BiasAdd_3Д
lstm_101/lstm_cell_101/mulMullstm_101/zeros:output:0(lstm_101/lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mulК
lstm_101/lstm_cell_101/mul_1Mullstm_101/zeros:output:0*lstm_101/lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_1К
lstm_101/lstm_cell_101/mul_2Mullstm_101/zeros:output:0*lstm_101/lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_2К
lstm_101/lstm_cell_101/mul_3Mullstm_101/zeros:output:0*lstm_101/lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_3О
%lstm_101/lstm_cell_101/ReadVariableOpReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_101/lstm_cell_101/ReadVariableOpЉ
*lstm_101/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_101/lstm_cell_101/strided_slice/stack­
,lstm_101/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_101/lstm_cell_101/strided_slice/stack_1­
,lstm_101/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_101/lstm_cell_101/strided_slice/stack_2
$lstm_101/lstm_cell_101/strided_sliceStridedSlice-lstm_101/lstm_cell_101/ReadVariableOp:value:03lstm_101/lstm_cell_101/strided_slice/stack:output:05lstm_101/lstm_cell_101/strided_slice/stack_1:output:05lstm_101/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_101/lstm_cell_101/strided_sliceЭ
lstm_101/lstm_cell_101/MatMul_4MatMullstm_101/lstm_cell_101/mul:z:0-lstm_101/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_4Ч
lstm_101/lstm_cell_101/addAddV2'lstm_101/lstm_cell_101/BiasAdd:output:0)lstm_101/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add
lstm_101/lstm_cell_101/SigmoidSigmoidlstm_101/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_101/lstm_cell_101/SigmoidТ
'lstm_101/lstm_cell_101/ReadVariableOp_1ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_1­
,lstm_101/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_101/lstm_cell_101/strided_slice_1/stackБ
.lstm_101/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_101/lstm_cell_101/strided_slice_1/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_1/stack_2
&lstm_101/lstm_cell_101/strided_slice_1StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_1:value:05lstm_101/lstm_cell_101/strided_slice_1/stack:output:07lstm_101/lstm_cell_101/strided_slice_1/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_1б
lstm_101/lstm_cell_101/MatMul_5MatMul lstm_101/lstm_cell_101/mul_1:z:0/lstm_101/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_5Э
lstm_101/lstm_cell_101/add_1AddV2)lstm_101/lstm_cell_101/BiasAdd_1:output:0)lstm_101/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_1Ѓ
 lstm_101/lstm_cell_101/Sigmoid_1Sigmoid lstm_101/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/Sigmoid_1Ж
lstm_101/lstm_cell_101/mul_4Mul$lstm_101/lstm_cell_101/Sigmoid_1:y:0lstm_101/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_4Т
'lstm_101/lstm_cell_101/ReadVariableOp_2ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_2­
,lstm_101/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_101/lstm_cell_101/strided_slice_2/stackБ
.lstm_101/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_101/lstm_cell_101/strided_slice_2/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_2/stack_2
&lstm_101/lstm_cell_101/strided_slice_2StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_2:value:05lstm_101/lstm_cell_101/strided_slice_2/stack:output:07lstm_101/lstm_cell_101/strided_slice_2/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_2б
lstm_101/lstm_cell_101/MatMul_6MatMul lstm_101/lstm_cell_101/mul_2:z:0/lstm_101/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_6Э
lstm_101/lstm_cell_101/add_2AddV2)lstm_101/lstm_cell_101/BiasAdd_2:output:0)lstm_101/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_2
lstm_101/lstm_cell_101/ReluRelu lstm_101/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/ReluФ
lstm_101/lstm_cell_101/mul_5Mul"lstm_101/lstm_cell_101/Sigmoid:y:0)lstm_101/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_5Л
lstm_101/lstm_cell_101/add_3AddV2 lstm_101/lstm_cell_101/mul_4:z:0 lstm_101/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_3Т
'lstm_101/lstm_cell_101/ReadVariableOp_3ReadVariableOp.lstm_101_lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02)
'lstm_101/lstm_cell_101/ReadVariableOp_3­
,lstm_101/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_101/lstm_cell_101/strided_slice_3/stackБ
.lstm_101/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_101/lstm_cell_101/strided_slice_3/stack_1Б
.lstm_101/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_101/lstm_cell_101/strided_slice_3/stack_2
&lstm_101/lstm_cell_101/strided_slice_3StridedSlice/lstm_101/lstm_cell_101/ReadVariableOp_3:value:05lstm_101/lstm_cell_101/strided_slice_3/stack:output:07lstm_101/lstm_cell_101/strided_slice_3/stack_1:output:07lstm_101/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_101/lstm_cell_101/strided_slice_3б
lstm_101/lstm_cell_101/MatMul_7MatMul lstm_101/lstm_cell_101/mul_3:z:0/lstm_101/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_101/lstm_cell_101/MatMul_7Э
lstm_101/lstm_cell_101/add_4AddV2)lstm_101/lstm_cell_101/BiasAdd_3:output:0)lstm_101/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/add_4Ѓ
 lstm_101/lstm_cell_101/Sigmoid_2Sigmoid lstm_101/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_101/lstm_cell_101/Sigmoid_2
lstm_101/lstm_cell_101/Relu_1Relu lstm_101/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/Relu_1Ш
lstm_101/lstm_cell_101/mul_6Mul$lstm_101/lstm_cell_101/Sigmoid_2:y:0+lstm_101/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_101/lstm_cell_101/mul_6Ё
&lstm_101/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&lstm_101/TensorArrayV2_1/element_shapeм
lstm_101/TensorArrayV2_1TensorListReserve/lstm_101/TensorArrayV2_1/element_shape:output:0!lstm_101/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_101/TensorArrayV2_1`
lstm_101/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/time
!lstm_101/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!lstm_101/while/maximum_iterations|
lstm_101/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_101/while/loop_counter
lstm_101/whileWhile$lstm_101/while/loop_counter:output:0*lstm_101/while/maximum_iterations:output:0lstm_101/time:output:0!lstm_101/TensorArrayV2_1:handle:0lstm_101/zeros:output:0lstm_101/zeros_1:output:0!lstm_101/strided_slice_1:output:0@lstm_101/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_101_lstm_cell_101_split_readvariableop_resource6lstm_101_lstm_cell_101_split_1_readvariableop_resource.lstm_101_lstm_cell_101_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_101_while_body_3239814*'
condR
lstm_101_while_cond_3239813*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_101/whileЧ
9lstm_101/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2;
9lstm_101/TensorArrayV2Stack/TensorListStack/element_shape
+lstm_101/TensorArrayV2Stack/TensorListStackTensorListStacklstm_101/while:output:3Blstm_101/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02-
+lstm_101/TensorArrayV2Stack/TensorListStack
lstm_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
lstm_101/strided_slice_3/stack
 lstm_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_101/strided_slice_3/stack_1
 lstm_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_101/strided_slice_3/stack_2а
lstm_101/strided_slice_3StridedSlice4lstm_101/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_101/strided_slice_3/stack:output:0)lstm_101/strided_slice_3/stack_1:output:0)lstm_101/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_101/strided_slice_3
lstm_101/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_101/transpose_1/permЩ
lstm_101/transpose_1	Transpose4lstm_101/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_101/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_101/transpose_1x
lstm_101/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_101/runtimeЋ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense_122/MatMul/ReadVariableOpЌ
dense_122/MatMulMatMul!lstm_101/strided_slice_3:output:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/MatMulЊ
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_122/BiasAdd/ReadVariableOpЉ
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/BiasAddv
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_122/ReluЋ
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense_123/MatMul/ReadVariableOpЇ
dense_123/MatMulMatMuldense_122/Relu:activations:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_123/MatMulЊ
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_123/BiasAdd/ReadVariableOpЉ
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_123/BiasAddn
reshape_61/ShapeShapedense_123/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_61/Shape
reshape_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_61/strided_slice/stack
 reshape_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_61/strided_slice/stack_1
 reshape_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_61/strided_slice/stack_2Є
reshape_61/strided_sliceStridedSlicereshape_61/Shape:output:0'reshape_61/strided_slice/stack:output:0)reshape_61/strided_slice/stack_1:output:0)reshape_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_61/strided_slicez
reshape_61/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_61/Reshape/shape/1z
reshape_61/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_61/Reshape/shape/2з
reshape_61/Reshape/shapePack!reshape_61/strided_slice:output:0#reshape_61/Reshape/shape/1:output:0#reshape_61/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_61/Reshape/shapeЈ
reshape_61/ReshapeReshapedense_123/BiasAdd:output:0!reshape_61/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_61/Reshapeј
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_101_lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulЪ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mulz
IdentityIdentityreshape_61/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityт
NoOpNoOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp1^dense_123/bias/Regularizer/Square/ReadVariableOp&^lstm_101/lstm_cell_101/ReadVariableOp(^lstm_101/lstm_cell_101/ReadVariableOp_1(^lstm_101/lstm_cell_101/ReadVariableOp_2(^lstm_101/lstm_cell_101/ReadVariableOp_3@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp,^lstm_101/lstm_cell_101/split/ReadVariableOp.^lstm_101/lstm_cell_101/split_1/ReadVariableOp^lstm_101/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2N
%lstm_101/lstm_cell_101/ReadVariableOp%lstm_101/lstm_cell_101/ReadVariableOp2R
'lstm_101/lstm_cell_101/ReadVariableOp_1'lstm_101/lstm_cell_101/ReadVariableOp_12R
'lstm_101/lstm_cell_101/ReadVariableOp_2'lstm_101/lstm_cell_101/ReadVariableOp_22R
'lstm_101/lstm_cell_101/ReadVariableOp_3'lstm_101/lstm_cell_101/ReadVariableOp_32
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2Z
+lstm_101/lstm_cell_101/split/ReadVariableOp+lstm_101/lstm_cell_101/split/ReadVariableOp2^
-lstm_101/lstm_cell_101/split_1/ReadVariableOp-lstm_101/lstm_cell_101/split_1/ReadVariableOp2 
lstm_101/whilelstm_101/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_3240991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3240991___redundant_placeholder05
1while_while_cond_3240991___redundant_placeholder15
1while_while_cond_3240991___redundant_placeholder25
1while_while_cond_3240991___redundant_placeholder3
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

Ќ
__inference_loss_fn_0_3241237G
9dense_123_bias_regularizer_square_readvariableop_resource:
identityЂ0dense_123/bias/Regularizer/Square/ReadVariableOpк
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOp9dense_123_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mull
IdentityIdentity"dense_123/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp1^dense_123/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp
ѕ

+__inference_dense_123_layer_call_fn_3241192

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallі
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
GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_32387592
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
УЕ
Б	
while_body_3238991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
!while/lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2#
!while/lstm_cell_101/dropout/ConstЯ
while/lstm_cell_101/dropout/MulMul&while/lstm_cell_101/ones_like:output:0*while/lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_101/dropout/Mul
!while/lstm_cell_101/dropout/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_101/dropout/Shape
8while/lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2сдт2:
8while/lstm_cell_101/dropout/random_uniform/RandomUniform
*while/lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2,
*while/lstm_cell_101/dropout/GreaterEqual/y
(while/lstm_cell_101/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_101/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_101/dropout/GreaterEqualЛ
 while/lstm_cell_101/dropout/CastCast,while/lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_101/dropout/CastЪ
!while/lstm_cell_101/dropout/Mul_1Mul#while/lstm_cell_101/dropout/Mul:z:0$while/lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout/Mul_1
#while/lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_1/Constе
!while/lstm_cell_101/dropout_1/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_1/Mul 
#while/lstm_cell_101/dropout_1/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_1/Shape
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ИЌ2<
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_1/GreaterEqual/y
*while/lstm_cell_101/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_1/GreaterEqualС
"while/lstm_cell_101/dropout_1/CastCast.while/lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_1/Castв
#while/lstm_cell_101/dropout_1/Mul_1Mul%while/lstm_cell_101/dropout_1/Mul:z:0&while/lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_1/Mul_1
#while/lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_2/Constе
!while/lstm_cell_101/dropout_2/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_2/Mul 
#while/lstm_cell_101/dropout_2/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_2/Shape
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ќ2<
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_2/GreaterEqual/y
*while/lstm_cell_101/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_2/GreaterEqualС
"while/lstm_cell_101/dropout_2/CastCast.while/lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_2/Castв
#while/lstm_cell_101/dropout_2/Mul_1Mul%while/lstm_cell_101/dropout_2/Mul:z:0&while/lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_2/Mul_1
#while/lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_3/Constе
!while/lstm_cell_101/dropout_3/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_3/Mul 
#while/lstm_cell_101/dropout_3/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_3/Shape
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЖІж2<
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_3/GreaterEqual/y
*while/lstm_cell_101/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_3/GreaterEqualС
"while/lstm_cell_101/dropout_3/CastCast.while/lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_3/Castв
#while/lstm_cell_101/dropout_3/Mul_1Mul%while/lstm_cell_101/dropout_3/Mul:z:0&while/lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_3/Mul_1
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ї
while/lstm_cell_101/mulMulwhile_placeholder_2%while/lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul­
while/lstm_cell_101/mul_1Mulwhile_placeholder_2'while/lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1­
while/lstm_cell_101/mul_2Mulwhile_placeholder_2'while/lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2­
while/lstm_cell_101/mul_3Mulwhile_placeholder_2'while/lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
while_cond_3237822
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3237822___redundant_placeholder05
1while_while_cond_3237822___redundant_placeholder15
1while_while_cond_3237822___redundant_placeholder25
1while_while_cond_3237822___redundant_placeholder3
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

Њ
F__inference_dense_123_layer_call_and_return_conditional_losses_3238759

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_123/bias/Regularizer/Square/ReadVariableOp
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
BiasAddР
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_123/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Т
Й
*__inference_lstm_101_layer_call_fn_3240035
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32381952
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
к
Ш
while_cond_3240716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3240716___redundant_placeholder05
1while_while_cond_3240716___redundant_placeholder15
1while_while_cond_3240716___redundant_placeholder25
1while_while_cond_3240716___redundant_placeholder3
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
Њ
З
*__inference_lstm_101_layer_call_fn_3240057

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32391562
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
р,
Т
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239324
input_42#
lstm_101_3239293:	
lstm_101_3239295:	#
lstm_101_3239297:	 #
dense_122_3239300:  
dense_122_3239302: #
dense_123_3239305: 
dense_123_3239307:
identityЂ!dense_122/StatefulPartitionedCallЂ!dense_123/StatefulPartitionedCallЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ lstm_101/StatefulPartitionedCallЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp­
 lstm_101/StatefulPartitionedCallStatefulPartitionedCallinput_42lstm_101_3239293lstm_101_3239295lstm_101_3239297*
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32391562"
 lstm_101/StatefulPartitionedCallП
!dense_122/StatefulPartitionedCallStatefulPartitionedCall)lstm_101/StatefulPartitionedCall:output:0dense_122_3239300dense_122_3239302*
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
GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_32387372#
!dense_122/StatefulPartitionedCallР
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_3239305dense_123_3239307*
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
GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_32387592#
!dense_123/StatefulPartitionedCall
reshape_61/PartitionedCallPartitionedCall*dense_123/StatefulPartitionedCall:output:0*
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_32387782
reshape_61/PartitionedCallд
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_101_3239293*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulВ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_3239307*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mul
IdentityIdentity#reshape_61/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall1^dense_123/bias/Regularizer/Square/ReadVariableOp!^lstm_101/StatefulPartitionedCall@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2D
 lstm_101/StatefulPartitionedCall lstm_101/StatefulPartitionedCall2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_42
ТЕ
Б	
while_body_3240992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
!while/lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2#
!while/lstm_cell_101/dropout/ConstЯ
while/lstm_cell_101/dropout/MulMul&while/lstm_cell_101/ones_like:output:0*while/lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_101/dropout/Mul
!while/lstm_cell_101/dropout/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_101/dropout/Shape
8while/lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform*while/lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2іДХ2:
8while/lstm_cell_101/dropout/random_uniform/RandomUniform
*while/lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2,
*while/lstm_cell_101/dropout/GreaterEqual/y
(while/lstm_cell_101/dropout/GreaterEqualGreaterEqualAwhile/lstm_cell_101/dropout/random_uniform/RandomUniform:output:03while/lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(while/lstm_cell_101/dropout/GreaterEqualЛ
 while/lstm_cell_101/dropout/CastCast,while/lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_101/dropout/CastЪ
!while/lstm_cell_101/dropout/Mul_1Mul#while/lstm_cell_101/dropout/Mul:z:0$while/lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout/Mul_1
#while/lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_1/Constе
!while/lstm_cell_101/dropout_1/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_1/Mul 
#while/lstm_cell_101/dropout_1/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_1/Shape
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Њщ2<
:while/lstm_cell_101/dropout_1/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_1/GreaterEqual/y
*while/lstm_cell_101/dropout_1/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_1/GreaterEqualС
"while/lstm_cell_101/dropout_1/CastCast.while/lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_1/Castв
#while/lstm_cell_101/dropout_1/Mul_1Mul%while/lstm_cell_101/dropout_1/Mul:z:0&while/lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_1/Mul_1
#while/lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_2/Constе
!while/lstm_cell_101/dropout_2/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_2/Mul 
#while/lstm_cell_101/dropout_2/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_2/Shape
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Є?2<
:while/lstm_cell_101/dropout_2/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_2/GreaterEqual/y
*while/lstm_cell_101/dropout_2/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_2/GreaterEqualС
"while/lstm_cell_101/dropout_2/CastCast.while/lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_2/Castв
#while/lstm_cell_101/dropout_2/Mul_1Mul%while/lstm_cell_101/dropout_2/Mul:z:0&while/lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_2/Mul_1
#while/lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2%
#while/lstm_cell_101/dropout_3/Constе
!while/lstm_cell_101/dropout_3/MulMul&while/lstm_cell_101/ones_like:output:0,while/lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_101/dropout_3/Mul 
#while/lstm_cell_101/dropout_3/ShapeShape&while/lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2%
#while/lstm_cell_101/dropout_3/Shape
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform,while/lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЛЈВ2<
:while/lstm_cell_101/dropout_3/random_uniform/RandomUniformЁ
,while/lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2.
,while/lstm_cell_101/dropout_3/GreaterEqual/y
*while/lstm_cell_101/dropout_3/GreaterEqualGreaterEqualCwhile/lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:05while/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*while/lstm_cell_101/dropout_3/GreaterEqualС
"while/lstm_cell_101/dropout_3/CastCast.while/lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_101/dropout_3/Castв
#while/lstm_cell_101/dropout_3/Mul_1Mul%while/lstm_cell_101/dropout_3/Mul:z:0&while/lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#while/lstm_cell_101/dropout_3/Mul_1
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ї
while/lstm_cell_101/mulMulwhile_placeholder_2%while/lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul­
while/lstm_cell_101/mul_1Mulwhile_placeholder_2'while/lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1­
while/lstm_cell_101/mul_2Mulwhile_placeholder_2'while/lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2­
while/lstm_cell_101/mul_3Mulwhile_placeholder_2'while/lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
щ

ќ
lstm_101_while_cond_3239510.
*lstm_101_while_lstm_101_while_loop_counter4
0lstm_101_while_lstm_101_while_maximum_iterations
lstm_101_while_placeholder 
lstm_101_while_placeholder_1 
lstm_101_while_placeholder_2 
lstm_101_while_placeholder_30
,lstm_101_while_less_lstm_101_strided_slice_1G
Clstm_101_while_lstm_101_while_cond_3239510___redundant_placeholder0G
Clstm_101_while_lstm_101_while_cond_3239510___redundant_placeholder1G
Clstm_101_while_lstm_101_while_cond_3239510___redundant_placeholder2G
Clstm_101_while_lstm_101_while_cond_3239510___redundant_placeholder3
lstm_101_while_identity

lstm_101/while/LessLesslstm_101_while_placeholder,lstm_101_while_less_lstm_101_strided_slice_1*
T0*
_output_shapes
: 2
lstm_101/while/Lessx
lstm_101/while/IdentityIdentitylstm_101/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_101/while/Identity";
lstm_101_while_identity lstm_101/while/Identity:output:0*(
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

c
G__inference_reshape_61_layer_call_and_return_conditional_losses_3238778

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
зR
а
E__inference_lstm_101_layer_call_and_return_conditional_losses_3238195

inputs(
lstm_cell_101_3238107:	$
lstm_cell_101_3238109:	(
lstm_cell_101_3238111:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂ%lstm_cell_101/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ї
%lstm_cell_101/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_101_3238107lstm_cell_101_3238109lstm_cell_101_3238111*
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
GPU 2J 8 *S
fNRL
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_32380422'
%lstm_cell_101/StatefulPartitionedCall
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
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_101_3238107lstm_cell_101_3238109lstm_cell_101_3238111*
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
while_body_3238120*
condR
while_cond_3238119*K
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
runtimeй
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_101_3238107*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityР
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp&^lstm_cell_101/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2N
%lstm_cell_101/StatefulPartitionedCall%lstm_cell_101/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъЃ
Д
E__inference_lstm_101_layer_call_and_return_conditional_losses_3238718

inputs>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileD
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0 lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3238585*
condR
while_cond_3238584*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Б	
while_body_3238585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ј
while/lstm_cell_101/mulMulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mulЌ
while/lstm_cell_101/mul_1Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1Ќ
while/lstm_cell_101/mul_2Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2Ќ
while/lstm_cell_101/mul_3Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
ўв
Ж
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240607
inputs_0>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileF
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout/ConstЗ
lstm_cell_101/dropout/MulMul lstm_cell_101/ones_like:output:0$lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul
lstm_cell_101/dropout/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout/Shapeњ
2lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2жУ024
2lstm_cell_101/dropout/random_uniform/RandomUniform
$lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_101/dropout/GreaterEqual/yі
"lstm_cell_101/dropout/GreaterEqualGreaterEqual;lstm_cell_101/dropout/random_uniform/RandomUniform:output:0-lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_101/dropout/GreaterEqualЉ
lstm_cell_101/dropout/CastCast&lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/CastВ
lstm_cell_101/dropout/Mul_1Mullstm_cell_101/dropout/Mul:z:0lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul_1
lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_1/ConstН
lstm_cell_101/dropout_1/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul
lstm_cell_101/dropout_1/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_1/Shape
4lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2еКБ26
4lstm_cell_101/dropout_1/random_uniform/RandomUniform
&lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_1/GreaterEqual/yў
$lstm_cell_101/dropout_1/GreaterEqualGreaterEqual=lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_1/GreaterEqualЏ
lstm_cell_101/dropout_1/CastCast(lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/CastК
lstm_cell_101/dropout_1/Mul_1Mullstm_cell_101/dropout_1/Mul:z:0 lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul_1
lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_2/ConstН
lstm_cell_101/dropout_2/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul
lstm_cell_101/dropout_2/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_2/Shape
4lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ёЏА26
4lstm_cell_101/dropout_2/random_uniform/RandomUniform
&lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_2/GreaterEqual/yў
$lstm_cell_101/dropout_2/GreaterEqualGreaterEqual=lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_2/GreaterEqualЏ
lstm_cell_101/dropout_2/CastCast(lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/CastК
lstm_cell_101/dropout_2/Mul_1Mullstm_cell_101/dropout_2/Mul:z:0 lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul_1
lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_3/ConstН
lstm_cell_101/dropout_3/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul
lstm_cell_101/dropout_3/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_3/Shape
4lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2иЈ26
4lstm_cell_101/dropout_3/random_uniform/RandomUniform
&lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_3/GreaterEqual/yў
$lstm_cell_101/dropout_3/GreaterEqualGreaterEqual=lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_3/GreaterEqualЏ
lstm_cell_101/dropout_3/CastCast(lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/CastК
lstm_cell_101/dropout_3/Mul_1Mullstm_cell_101/dropout_3/Mul:z:0 lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul_1
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0!lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0!lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0!lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3240442*
condR
while_cond_3240441*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

Б	
while_body_3240717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_101_split_readvariableop_resource_0:	D
5while_lstm_cell_101_split_1_readvariableop_resource_0:	@
-while_lstm_cell_101_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_101_split_readvariableop_resource:	B
3while_lstm_cell_101_split_1_readvariableop_resource:	>
+while_lstm_cell_101_readvariableop_resource:	 Ђ"while/lstm_cell_101/ReadVariableOpЂ$while/lstm_cell_101/ReadVariableOp_1Ђ$while/lstm_cell_101/ReadVariableOp_2Ђ$while/lstm_cell_101/ReadVariableOp_3Ђ(while/lstm_cell_101/split/ReadVariableOpЂ*while/lstm_cell_101/split_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItem
#while/lstm_cell_101/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_101/ones_like/Shape
#while/lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_101/ones_like/Constд
while/lstm_cell_101/ones_likeFill,while/lstm_cell_101/ones_like/Shape:output:0,while/lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ones_like
#while/lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_101/split/split_dimЩ
(while/lstm_cell_101/split/ReadVariableOpReadVariableOp3while_lstm_cell_101_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_101/split/ReadVariableOpї
while/lstm_cell_101/splitSplit,while/lstm_cell_101/split/split_dim:output:00while/lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_101/splitЪ
while/lstm_cell_101/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMulЮ
while/lstm_cell_101/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_1Ю
while/lstm_cell_101/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_2Ю
while/lstm_cell_101/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_3
%while/lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%while/lstm_cell_101/split_1/split_dimЫ
*while/lstm_cell_101/split_1/ReadVariableOpReadVariableOp5while_lstm_cell_101_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02,
*while/lstm_cell_101/split_1/ReadVariableOpя
while/lstm_cell_101/split_1Split.while/lstm_cell_101/split_1/split_dim:output:02while/lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_101/split_1У
while/lstm_cell_101/BiasAddBiasAdd$while/lstm_cell_101/MatMul:product:0$while/lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAddЩ
while/lstm_cell_101/BiasAdd_1BiasAdd&while/lstm_cell_101/MatMul_1:product:0$while/lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_1Щ
while/lstm_cell_101/BiasAdd_2BiasAdd&while/lstm_cell_101/MatMul_2:product:0$while/lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_2Щ
while/lstm_cell_101/BiasAdd_3BiasAdd&while/lstm_cell_101/MatMul_3:product:0$while/lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/BiasAdd_3Ј
while/lstm_cell_101/mulMulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mulЌ
while/lstm_cell_101/mul_1Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_1Ќ
while/lstm_cell_101/mul_2Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_2Ќ
while/lstm_cell_101/mul_3Mulwhile_placeholder_2&while/lstm_cell_101/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_3З
"while/lstm_cell_101/ReadVariableOpReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02$
"while/lstm_cell_101/ReadVariableOpЃ
'while/lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_101/strided_slice/stackЇ
)while/lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice/stack_1Ї
)while/lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_101/strided_slice/stack_2є
!while/lstm_cell_101/strided_sliceStridedSlice*while/lstm_cell_101/ReadVariableOp:value:00while/lstm_cell_101/strided_slice/stack:output:02while/lstm_cell_101/strided_slice/stack_1:output:02while/lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_101/strided_sliceС
while/lstm_cell_101/MatMul_4MatMulwhile/lstm_cell_101/mul:z:0*while/lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_4Л
while/lstm_cell_101/addAddV2$while/lstm_cell_101/BiasAdd:output:0&while/lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add
while/lstm_cell_101/SigmoidSigmoidwhile/lstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/SigmoidЛ
$while/lstm_cell_101/ReadVariableOp_1ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_1Ї
)while/lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_101/strided_slice_1/stackЋ
+while/lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+while/lstm_cell_101/strided_slice_1/stack_1Ћ
+while/lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_1/stack_2
#while/lstm_cell_101/strided_slice_1StridedSlice,while/lstm_cell_101/ReadVariableOp_1:value:02while/lstm_cell_101/strided_slice_1/stack:output:04while/lstm_cell_101/strided_slice_1/stack_1:output:04while/lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_1Х
while/lstm_cell_101/MatMul_5MatMulwhile/lstm_cell_101/mul_1:z:0,while/lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_5С
while/lstm_cell_101/add_1AddV2&while/lstm_cell_101/BiasAdd_1:output:0&while/lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_1
while/lstm_cell_101/Sigmoid_1Sigmoidwhile/lstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_1Ї
while/lstm_cell_101/mul_4Mul!while/lstm_cell_101/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_4Л
$while/lstm_cell_101/ReadVariableOp_2ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_2Ї
)while/lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_101/strided_slice_2/stackЋ
+while/lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+while/lstm_cell_101/strided_slice_2/stack_1Ћ
+while/lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_2/stack_2
#while/lstm_cell_101/strided_slice_2StridedSlice,while/lstm_cell_101/ReadVariableOp_2:value:02while/lstm_cell_101/strided_slice_2/stack:output:04while/lstm_cell_101/strided_slice_2/stack_1:output:04while/lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_2Х
while/lstm_cell_101/MatMul_6MatMulwhile/lstm_cell_101/mul_2:z:0,while/lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_6С
while/lstm_cell_101/add_2AddV2&while/lstm_cell_101/BiasAdd_2:output:0&while/lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_2
while/lstm_cell_101/ReluReluwhile/lstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/ReluИ
while/lstm_cell_101/mul_5Mulwhile/lstm_cell_101/Sigmoid:y:0&while/lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_5Џ
while/lstm_cell_101/add_3AddV2while/lstm_cell_101/mul_4:z:0while/lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_3Л
$while/lstm_cell_101/ReadVariableOp_3ReadVariableOp-while_lstm_cell_101_readvariableop_resource_0*
_output_shapes
:	 *
dtype02&
$while/lstm_cell_101/ReadVariableOp_3Ї
)while/lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_101/strided_slice_3/stackЋ
+while/lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+while/lstm_cell_101/strided_slice_3/stack_1Ћ
+while/lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+while/lstm_cell_101/strided_slice_3/stack_2
#while/lstm_cell_101/strided_slice_3StridedSlice,while/lstm_cell_101/ReadVariableOp_3:value:02while/lstm_cell_101/strided_slice_3/stack:output:04while/lstm_cell_101/strided_slice_3/stack_1:output:04while/lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#while/lstm_cell_101/strided_slice_3Х
while/lstm_cell_101/MatMul_7MatMulwhile/lstm_cell_101/mul_3:z:0,while/lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/MatMul_7С
while/lstm_cell_101/add_4AddV2&while/lstm_cell_101/BiasAdd_3:output:0&while/lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/add_4
while/lstm_cell_101/Sigmoid_2Sigmoidwhile/lstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Sigmoid_2
while/lstm_cell_101/Relu_1Reluwhile/lstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/Relu_1М
while/lstm_cell_101/mul_6Mul!while/lstm_cell_101/Sigmoid_2:y:0(while/lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_101/mul_6с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_101/mul_6:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_101/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_101/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ь

while/NoOpNoOp#^while/lstm_cell_101/ReadVariableOp%^while/lstm_cell_101/ReadVariableOp_1%^while/lstm_cell_101/ReadVariableOp_2%^while/lstm_cell_101/ReadVariableOp_3)^while/lstm_cell_101/split/ReadVariableOp+^while/lstm_cell_101/split_1/ReadVariableOp*"
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
+while_lstm_cell_101_readvariableop_resource-while_lstm_cell_101_readvariableop_resource_0"l
3while_lstm_cell_101_split_1_readvariableop_resource5while_lstm_cell_101_split_1_readvariableop_resource_0"h
1while_lstm_cell_101_split_readvariableop_resource3while_lstm_cell_101_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2H
"while/lstm_cell_101/ReadVariableOp"while/lstm_cell_101/ReadVariableOp2L
$while/lstm_cell_101/ReadVariableOp_1$while/lstm_cell_101/ReadVariableOp_12L
$while/lstm_cell_101/ReadVariableOp_2$while/lstm_cell_101/ReadVariableOp_22L
$while/lstm_cell_101/ReadVariableOp_3$while/lstm_cell_101/ReadVariableOp_32T
(while/lstm_cell_101/split/ReadVariableOp(while/lstm_cell_101/split/ReadVariableOp2X
*while/lstm_cell_101/split_1/ReadVariableOp*while/lstm_cell_101/split_1/ReadVariableOp: 
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
Шв
Д
E__inference_lstm_101_layer_call_and_return_conditional_losses_3239156

inputs>
+lstm_cell_101_split_readvariableop_resource:	<
-lstm_cell_101_split_1_readvariableop_resource:	8
%lstm_cell_101_readvariableop_resource:	 
identityЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_101/ReadVariableOpЂlstm_cell_101/ReadVariableOp_1Ђlstm_cell_101/ReadVariableOp_2Ђlstm_cell_101/ReadVariableOp_3Ђ"lstm_cell_101/split/ReadVariableOpЂ$lstm_cell_101/split_1/ReadVariableOpЂwhileD
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
strided_slice_2|
lstm_cell_101/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_101/ones_like/Shape
lstm_cell_101/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_101/ones_like/ConstМ
lstm_cell_101/ones_likeFill&lstm_cell_101/ones_like/Shape:output:0&lstm_cell_101/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/ones_like
lstm_cell_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout/ConstЗ
lstm_cell_101/dropout/MulMul lstm_cell_101/ones_like:output:0$lstm_cell_101/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul
lstm_cell_101/dropout/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout/Shapeћ
2lstm_cell_101/dropout/random_uniform/RandomUniformRandomUniform$lstm_cell_101/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ю24
2lstm_cell_101/dropout/random_uniform/RandomUniform
$lstm_cell_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_101/dropout/GreaterEqual/yі
"lstm_cell_101/dropout/GreaterEqualGreaterEqual;lstm_cell_101/dropout/random_uniform/RandomUniform:output:0-lstm_cell_101/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_cell_101/dropout/GreaterEqualЉ
lstm_cell_101/dropout/CastCast&lstm_cell_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/CastВ
lstm_cell_101/dropout/Mul_1Mullstm_cell_101/dropout/Mul:z:0lstm_cell_101/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout/Mul_1
lstm_cell_101/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_1/ConstН
lstm_cell_101/dropout_1/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul
lstm_cell_101/dropout_1/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_1/Shape
4lstm_cell_101/dropout_1/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЭЙ26
4lstm_cell_101/dropout_1/random_uniform/RandomUniform
&lstm_cell_101/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_1/GreaterEqual/yў
$lstm_cell_101/dropout_1/GreaterEqualGreaterEqual=lstm_cell_101/dropout_1/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_1/GreaterEqualЏ
lstm_cell_101/dropout_1/CastCast(lstm_cell_101/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/CastК
lstm_cell_101/dropout_1/Mul_1Mullstm_cell_101/dropout_1/Mul:z:0 lstm_cell_101/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_1/Mul_1
lstm_cell_101/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_2/ConstН
lstm_cell_101/dropout_2/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul
lstm_cell_101/dropout_2/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_2/Shape
4lstm_cell_101/dropout_2/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ед№26
4lstm_cell_101/dropout_2/random_uniform/RandomUniform
&lstm_cell_101/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_2/GreaterEqual/yў
$lstm_cell_101/dropout_2/GreaterEqualGreaterEqual=lstm_cell_101/dropout_2/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_2/GreaterEqualЏ
lstm_cell_101/dropout_2/CastCast(lstm_cell_101/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/CastК
lstm_cell_101/dropout_2/Mul_1Mullstm_cell_101/dropout_2/Mul:z:0 lstm_cell_101/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_2/Mul_1
lstm_cell_101/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_101/dropout_3/ConstН
lstm_cell_101/dropout_3/MulMul lstm_cell_101/ones_like:output:0&lstm_cell_101/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul
lstm_cell_101/dropout_3/ShapeShape lstm_cell_101/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_101/dropout_3/Shape
4lstm_cell_101/dropout_3/random_uniform/RandomUniformRandomUniform&lstm_cell_101/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЋW26
4lstm_cell_101/dropout_3/random_uniform/RandomUniform
&lstm_cell_101/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2(
&lstm_cell_101/dropout_3/GreaterEqual/yў
$lstm_cell_101/dropout_3/GreaterEqualGreaterEqual=lstm_cell_101/dropout_3/random_uniform/RandomUniform:output:0/lstm_cell_101/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_cell_101/dropout_3/GreaterEqualЏ
lstm_cell_101/dropout_3/CastCast(lstm_cell_101/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/CastК
lstm_cell_101/dropout_3/Mul_1Mullstm_cell_101/dropout_3/Mul:z:0 lstm_cell_101/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/dropout_3/Mul_1
lstm_cell_101/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_101/split/split_dimЕ
"lstm_cell_101/split/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_101/split/ReadVariableOpп
lstm_cell_101/splitSplit&lstm_cell_101/split/split_dim:output:0*lstm_cell_101/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_101/split 
lstm_cell_101/MatMulMatMulstrided_slice_2:output:0lstm_cell_101/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMulЄ
lstm_cell_101/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_101/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_1Є
lstm_cell_101/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_101/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_2Є
lstm_cell_101/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_101/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_3
lstm_cell_101/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
lstm_cell_101/split_1/split_dimЗ
$lstm_cell_101/split_1/ReadVariableOpReadVariableOp-lstm_cell_101_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$lstm_cell_101/split_1/ReadVariableOpз
lstm_cell_101/split_1Split(lstm_cell_101/split_1/split_dim:output:0,lstm_cell_101/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_101/split_1Ћ
lstm_cell_101/BiasAddBiasAddlstm_cell_101/MatMul:product:0lstm_cell_101/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAddБ
lstm_cell_101/BiasAdd_1BiasAdd lstm_cell_101/MatMul_1:product:0lstm_cell_101/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_1Б
lstm_cell_101/BiasAdd_2BiasAdd lstm_cell_101/MatMul_2:product:0lstm_cell_101/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_2Б
lstm_cell_101/BiasAdd_3BiasAdd lstm_cell_101/MatMul_3:product:0lstm_cell_101/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/BiasAdd_3
lstm_cell_101/mulMulzeros:output:0lstm_cell_101/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul
lstm_cell_101/mul_1Mulzeros:output:0!lstm_cell_101/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_1
lstm_cell_101/mul_2Mulzeros:output:0!lstm_cell_101/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_2
lstm_cell_101/mul_3Mulzeros:output:0!lstm_cell_101/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_3Ѓ
lstm_cell_101/ReadVariableOpReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_101/ReadVariableOp
!lstm_cell_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_101/strided_slice/stack
#lstm_cell_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice/stack_1
#lstm_cell_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_101/strided_slice/stack_2а
lstm_cell_101/strided_sliceStridedSlice$lstm_cell_101/ReadVariableOp:value:0*lstm_cell_101/strided_slice/stack:output:0,lstm_cell_101/strided_slice/stack_1:output:0,lstm_cell_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_sliceЉ
lstm_cell_101/MatMul_4MatMullstm_cell_101/mul:z:0$lstm_cell_101/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_4Ѓ
lstm_cell_101/addAddV2lstm_cell_101/BiasAdd:output:0 lstm_cell_101/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add
lstm_cell_101/SigmoidSigmoidlstm_cell_101/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/SigmoidЇ
lstm_cell_101/ReadVariableOp_1ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_1
#lstm_cell_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_101/strided_slice_1/stack
%lstm_cell_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%lstm_cell_101/strided_slice_1/stack_1
%lstm_cell_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_1/stack_2м
lstm_cell_101/strided_slice_1StridedSlice&lstm_cell_101/ReadVariableOp_1:value:0,lstm_cell_101/strided_slice_1/stack:output:0.lstm_cell_101/strided_slice_1/stack_1:output:0.lstm_cell_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_1­
lstm_cell_101/MatMul_5MatMullstm_cell_101/mul_1:z:0&lstm_cell_101/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_5Љ
lstm_cell_101/add_1AddV2 lstm_cell_101/BiasAdd_1:output:0 lstm_cell_101/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_1
lstm_cell_101/Sigmoid_1Sigmoidlstm_cell_101/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_1
lstm_cell_101/mul_4Mullstm_cell_101/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_4Ї
lstm_cell_101/ReadVariableOp_2ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_2
#lstm_cell_101/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_101/strided_slice_2/stack
%lstm_cell_101/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_cell_101/strided_slice_2/stack_1
%lstm_cell_101/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_2/stack_2м
lstm_cell_101/strided_slice_2StridedSlice&lstm_cell_101/ReadVariableOp_2:value:0,lstm_cell_101/strided_slice_2/stack:output:0.lstm_cell_101/strided_slice_2/stack_1:output:0.lstm_cell_101/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_2­
lstm_cell_101/MatMul_6MatMullstm_cell_101/mul_2:z:0&lstm_cell_101/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_6Љ
lstm_cell_101/add_2AddV2 lstm_cell_101/BiasAdd_2:output:0 lstm_cell_101/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_2{
lstm_cell_101/ReluRelulstm_cell_101/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu 
lstm_cell_101/mul_5Mullstm_cell_101/Sigmoid:y:0 lstm_cell_101/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_5
lstm_cell_101/add_3AddV2lstm_cell_101/mul_4:z:0lstm_cell_101/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_3Ї
lstm_cell_101/ReadVariableOp_3ReadVariableOp%lstm_cell_101_readvariableop_resource*
_output_shapes
:	 *
dtype02 
lstm_cell_101/ReadVariableOp_3
#lstm_cell_101/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_101/strided_slice_3/stack
%lstm_cell_101/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%lstm_cell_101/strided_slice_3/stack_1
%lstm_cell_101/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_cell_101/strided_slice_3/stack_2м
lstm_cell_101/strided_slice_3StridedSlice&lstm_cell_101/ReadVariableOp_3:value:0,lstm_cell_101/strided_slice_3/stack:output:0.lstm_cell_101/strided_slice_3/stack_1:output:0.lstm_cell_101/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_101/strided_slice_3­
lstm_cell_101/MatMul_7MatMullstm_cell_101/mul_3:z:0&lstm_cell_101/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/MatMul_7Љ
lstm_cell_101/add_4AddV2 lstm_cell_101/BiasAdd_3:output:0 lstm_cell_101/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/add_4
lstm_cell_101/Sigmoid_2Sigmoidlstm_cell_101/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Sigmoid_2
lstm_cell_101/Relu_1Relulstm_cell_101/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/Relu_1Є
lstm_cell_101/mul_6Mullstm_cell_101/Sigmoid_2:y:0"lstm_cell_101/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_101/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_101_split_readvariableop_resource-lstm_cell_101_split_1_readvariableop_resource%lstm_cell_101_readvariableop_resource*
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
while_body_3238991*
condR
while_cond_3238990*K
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
runtimeя
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_101_split_readvariableop_resource*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityц
NoOpNoOp@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_101/ReadVariableOp^lstm_cell_101/ReadVariableOp_1^lstm_cell_101/ReadVariableOp_2^lstm_cell_101/ReadVariableOp_3#^lstm_cell_101/split/ReadVariableOp%^lstm_cell_101/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp2<
lstm_cell_101/ReadVariableOplstm_cell_101/ReadVariableOp2@
lstm_cell_101/ReadVariableOp_1lstm_cell_101/ReadVariableOp_12@
lstm_cell_101/ReadVariableOp_2lstm_cell_101/ReadVariableOp_22@
lstm_cell_101/ReadVariableOp_3lstm_cell_101/ReadVariableOp_32H
"lstm_cell_101/split/ReadVariableOp"lstm_cell_101/split/ReadVariableOp2L
$lstm_cell_101/split_1/ReadVariableOp$lstm_cell_101/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к,
Р
J__inference_sequential_41_layer_call_and_return_conditional_losses_3238793

inputs#
lstm_101_3238719:	
lstm_101_3238721:	#
lstm_101_3238723:	 #
dense_122_3238738:  
dense_122_3238740: #
dense_123_3238760: 
dense_123_3238762:
identityЂ!dense_122/StatefulPartitionedCallЂ!dense_123/StatefulPartitionedCallЂ0dense_123/bias/Regularizer/Square/ReadVariableOpЂ lstm_101/StatefulPartitionedCallЂ?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpЋ
 lstm_101/StatefulPartitionedCallStatefulPartitionedCallinputslstm_101_3238719lstm_101_3238721lstm_101_3238723*
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
GPU 2J 8 *N
fIRG
E__inference_lstm_101_layer_call_and_return_conditional_losses_32387182"
 lstm_101/StatefulPartitionedCallП
!dense_122/StatefulPartitionedCallStatefulPartitionedCall)lstm_101/StatefulPartitionedCall:output:0dense_122_3238738dense_122_3238740*
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
GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_32387372#
!dense_122/StatefulPartitionedCallР
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_3238760dense_123_3238762*
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
GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_32387592#
!dense_123/StatefulPartitionedCall
reshape_61/PartitionedCallPartitionedCall*dense_123/StatefulPartitionedCall:output:0*
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_32387782
reshape_61/PartitionedCallд
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_101_3238719*
_output_shapes
:	*
dtype02A
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOpс
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareSquareGlstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
0lstm_101/lstm_cell_101/kernel/Regularizer/SquareГ
/lstm_101/lstm_cell_101/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/lstm_101/lstm_cell_101/kernel/Regularizer/Constі
-lstm_101/lstm_cell_101/kernel/Regularizer/SumSum4lstm_101/lstm_cell_101/kernel/Regularizer/Square:y:08lstm_101/lstm_cell_101/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/SumЇ
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб821
/lstm_101/lstm_cell_101/kernel/Regularizer/mul/xј
-lstm_101/lstm_cell_101/kernel/Regularizer/mulMul8lstm_101/lstm_cell_101/kernel/Regularizer/mul/x:output:06lstm_101/lstm_cell_101/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-lstm_101/lstm_cell_101/kernel/Regularizer/mulВ
0dense_123/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_3238762*
_output_shapes
:*
dtype022
0dense_123/bias/Regularizer/Square/ReadVariableOpЏ
!dense_123/bias/Regularizer/SquareSquare8dense_123/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!dense_123/bias/Regularizer/Square
 dense_123/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dense_123/bias/Regularizer/ConstК
dense_123/bias/Regularizer/SumSum%dense_123/bias/Regularizer/Square:y:0)dense_123/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/Sum
 dense_123/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_123/bias/Regularizer/mul/xМ
dense_123/bias/Regularizer/mulMul)dense_123/bias/Regularizer/mul/x:output:0'dense_123/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_123/bias/Regularizer/mul
IdentityIdentity#reshape_61/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall1^dense_123/bias/Regularizer/Square/ReadVariableOp!^lstm_101/StatefulPartitionedCall@^lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2d
0dense_123/bias/Regularizer/Square/ReadVariableOp0dense_123/bias/Regularizer/Square/ReadVariableOp2D
 lstm_101/StatefulPartitionedCall lstm_101/StatefulPartitionedCall2
?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp?lstm_101/lstm_cell_101/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_3238119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3238119___redundant_placeholder05
1while_while_cond_3238119___redundant_placeholder15
1while_while_cond_3238119___redundant_placeholder25
1while_while_cond_3238119___redundant_placeholder3
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
:"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*З
serving_defaultЃ
A
input_425
serving_default_input_42:0џџџџџџџџџB

reshape_614
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
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
":   2dense_122/kernel
: 2dense_122/bias
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
":  2dense_123/kernel
:2dense_123/bias
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
0:.	2lstm_101/lstm_cell_101/kernel
::8	 2'lstm_101/lstm_cell_101/recurrent_kernel
*:(2lstm_101/lstm_cell_101/bias
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
':%  2Adam/dense_122/kernel/m
!: 2Adam/dense_122/bias/m
':% 2Adam/dense_123/kernel/m
!:2Adam/dense_123/bias/m
5:3	2$Adam/lstm_101/lstm_cell_101/kernel/m
?:=	 2.Adam/lstm_101/lstm_cell_101/recurrent_kernel/m
/:-2"Adam/lstm_101/lstm_cell_101/bias/m
':%  2Adam/dense_122/kernel/v
!: 2Adam/dense_122/bias/v
':% 2Adam/dense_123/kernel/v
!:2Adam/dense_123/bias/v
5:3	2$Adam/lstm_101/lstm_cell_101/kernel/v
?:=	 2.Adam/lstm_101/lstm_cell_101/recurrent_kernel/v
/:-2"Adam/lstm_101/lstm_cell_101/bias/v
ЮBЫ
"__inference__wrapped_model_3237685input_42"
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
/__inference_sequential_41_layer_call_fn_3238810
/__inference_sequential_41_layer_call_fn_3239382
/__inference_sequential_41_layer_call_fn_3239401
/__inference_sequential_41_layer_call_fn_3239256Р
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
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239672
J__inference_sequential_41_layer_call_and_return_conditional_losses_3240007
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239290
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239324Р
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
2
*__inference_lstm_101_layer_call_fn_3240024
*__inference_lstm_101_layer_call_fn_3240035
*__inference_lstm_101_layer_call_fn_3240046
*__inference_lstm_101_layer_call_fn_3240057е
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
ї2є
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240300
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240607
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240850
E__inference_lstm_101_layer_call_and_return_conditional_losses_3241157е
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
е2в
+__inference_dense_122_layer_call_fn_3241166Ђ
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
№2э
F__inference_dense_122_layer_call_and_return_conditional_losses_3241177Ђ
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
е2в
+__inference_dense_123_layer_call_fn_3241192Ђ
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
№2э
F__inference_dense_123_layer_call_and_return_conditional_losses_3241208Ђ
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
,__inference_reshape_61_layer_call_fn_3241213Ђ
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_3241226Ђ
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
__inference_loss_fn_0_3241237
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
%__inference_signature_wrapper_3239363input_42"
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
І2Ѓ
/__inference_lstm_cell_101_layer_call_fn_3241260
/__inference_lstm_cell_101_layer_call_fn_3241277О
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
м2й
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241358
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241471О
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
__inference_loss_fn_1_3241482
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
"__inference__wrapped_model_3237685}&('5Ђ2
+Ђ(
&#
input_42џџџџџџџџџ
Њ ";Њ8
6

reshape_61(%

reshape_61џџџџџџџџџІ
F__inference_dense_122_layer_call_and_return_conditional_losses_3241177\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 ~
+__inference_dense_122_layer_call_fn_3241166O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ І
F__inference_dense_123_layer_call_and_return_conditional_losses_3241208\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_123_layer_call_fn_3241192O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ<
__inference_loss_fn_0_3241237Ђ

Ђ 
Њ " <
__inference_loss_fn_1_3241482&Ђ

Ђ 
Њ " Ц
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240300}&('OЂL
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
 Ц
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240607}&('OЂL
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
 Ж
E__inference_lstm_101_layer_call_and_return_conditional_losses_3240850m&('?Ђ<
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
 Ж
E__inference_lstm_101_layer_call_and_return_conditional_losses_3241157m&('?Ђ<
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
 
*__inference_lstm_101_layer_call_fn_3240024p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
*__inference_lstm_101_layer_call_fn_3240035p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ 
*__inference_lstm_101_layer_call_fn_3240046`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
*__inference_lstm_101_layer_call_fn_3240057`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ Ь
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241358§&('Ђ}
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
 Ь
J__inference_lstm_cell_101_layer_call_and_return_conditional_losses_3241471§&('Ђ}
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
 Ё
/__inference_lstm_cell_101_layer_call_fn_3241260э&('Ђ}
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
1/1џџџџџџџџџ Ё
/__inference_lstm_cell_101_layer_call_fn_3241277э&('Ђ}
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
G__inference_reshape_61_layer_call_and_return_conditional_losses_3241226\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
,__inference_reshape_61_layer_call_fn_3241213O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџС
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239290s&('=Ђ:
3Ђ0
&#
input_42џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 С
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239324s&('=Ђ:
3Ђ0
&#
input_42џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 П
J__inference_sequential_41_layer_call_and_return_conditional_losses_3239672q&(';Ђ8
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
J__inference_sequential_41_layer_call_and_return_conditional_losses_3240007q&(';Ђ8
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
/__inference_sequential_41_layer_call_fn_3238810f&('=Ђ:
3Ђ0
&#
input_42џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_41_layer_call_fn_3239256f&('=Ђ:
3Ђ0
&#
input_42џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_41_layer_call_fn_3239382d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_41_layer_call_fn_3239401d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџГ
%__inference_signature_wrapper_3239363&('AЂ>
Ђ 
7Њ4
2
input_42&#
input_42џџџџџџџџџ";Њ8
6

reshape_61(%

reshape_61џџџџџџџџџ