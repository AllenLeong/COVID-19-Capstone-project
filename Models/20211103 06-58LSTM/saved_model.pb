Эу&
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8Х%
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:  *
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
: *
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

: *
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
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
lstm_15/lstm_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_15/lstm_cell_15/kernel

/lstm_15/lstm_cell_15/kernel/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/kernel*
_output_shapes
:	*
dtype0
Ї
%lstm_15/lstm_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%lstm_15/lstm_cell_15/recurrent_kernel
 
9lstm_15/lstm_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_15/lstm_cell_15/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_15/lstm_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_15/lstm_cell_15/bias

-lstm_15/lstm_cell_15/bias/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/bias*
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
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_18/kernel/m

*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
: *
dtype0

Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_19/kernel/m

*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_15/lstm_cell_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_15/lstm_cell_15/kernel/m

6Adam/lstm_15/lstm_cell_15/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_15/lstm_cell_15/kernel/m*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_15/lstm_cell_15/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_15/lstm_cell_15/recurrent_kernel/m
Ў
@Adam/lstm_15/lstm_cell_15/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_15/lstm_cell_15/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

 Adam/lstm_15/lstm_cell_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_15/lstm_cell_15/bias/m

4Adam/lstm_15/lstm_cell_15/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_15/lstm_cell_15/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_18/kernel/v

*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
: *
dtype0

Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_19/kernel/v

*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_15/lstm_cell_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_15/lstm_cell_15/kernel/v

6Adam/lstm_15/lstm_cell_15/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_15/lstm_cell_15/kernel/v*
_output_shapes
:	*
dtype0
Е
,Adam/lstm_15/lstm_cell_15/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_15/lstm_cell_15/recurrent_kernel/v
Ў
@Adam/lstm_15/lstm_cell_15/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_15/lstm_cell_15/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

 Adam/lstm_15/lstm_cell_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_15/lstm_cell_15/bias/v

4Adam/lstm_15/lstm_cell_15/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_15/lstm_cell_15/bias/v*
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
trainable_variables
)layer_metrics
	variables
*metrics

+layers
,non_trainable_variables
regularization_losses
-layer_regularization_losses
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
trainable_variables
3layer_metrics
	variables
4metrics

5layers

6states
7non_trainable_variables
regularization_losses
8layer_regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
9layer_metrics
:metrics
	variables

;layers
<non_trainable_variables
regularization_losses
=layer_regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
>layer_metrics
?metrics
	variables

@layers
Anon_trainable_variables
regularization_losses
Blayer_regularization_losses
 
 
 
­
trainable_variables
Clayer_metrics
Dmetrics
	variables

Elayers
Fnon_trainable_variables
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
a_
VARIABLE_VALUElstm_15/lstm_cell_15/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_15/lstm_cell_15/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_15/lstm_cell_15/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

H0

0
1
2
3
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
/trainable_variables
Ilayer_metrics
Jmetrics
0	variables

Klayers
Lnon_trainable_variables
1regularization_losses
Mlayer_regularization_losses
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
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_15/lstm_cell_15/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_15/lstm_cell_15/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_15/lstm_cell_15/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_15/lstm_cell_15/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_15/lstm_cell_15/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_15/lstm_cell_15/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_7Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
у
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7lstm_15/lstm_cell_15/kernellstm_15/lstm_cell_15/bias%lstm_15/lstm_cell_15/recurrent_kerneldense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
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
$__inference_signature_wrapper_606098
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
љ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_15/lstm_cell_15/kernel/Read/ReadVariableOp9lstm_15/lstm_cell_15/recurrent_kernel/Read/ReadVariableOp-lstm_15/lstm_cell_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp6Adam/lstm_15/lstm_cell_15/kernel/m/Read/ReadVariableOp@Adam/lstm_15/lstm_cell_15/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_15/lstm_cell_15/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp6Adam/lstm_15/lstm_cell_15/kernel/v/Read/ReadVariableOp@Adam/lstm_15/lstm_cell_15/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_15/lstm_cell_15/bias/v/Read/ReadVariableOpConst*)
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
__inference__traced_save_608324
Ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasdense_19/kerneldense_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_15/lstm_cell_15/kernel%lstm_15/lstm_cell_15/recurrent_kernellstm_15/lstm_cell_15/biastotalcountAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/m"Adam/lstm_15/lstm_cell_15/kernel/m,Adam/lstm_15/lstm_cell_15/recurrent_kernel/m Adam/lstm_15/lstm_cell_15/bias/mAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/v"Adam/lstm_15/lstm_cell_15/kernel/v,Adam/lstm_15/lstm_cell_15/recurrent_kernel/v Adam/lstm_15/lstm_cell_15/bias/v*(
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
"__inference__traced_restore_608418Ц$
е
У
while_cond_607407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_607407___redundant_placeholder04
0while_while_cond_607407___redundant_placeholder14
0while_while_cond_607407___redundant_placeholder24
0while_while_cond_607407___redundant_placeholder3
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
while_cond_605725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_605725___redundant_placeholder04
0while_while_cond_605725___redundant_placeholder14
0while_while_cond_605725___redundant_placeholder24
0while_while_cond_605725___redundant_placeholder3
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
)__inference_dense_18_layer_call_fn_607912

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
D__inference_dense_18_layer_call_and_return_conditional_losses_6054722
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
и%
у
while_body_604558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_15_604582_0:	*
while_lstm_cell_15_604584_0:	.
while_lstm_cell_15_604586_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_15_604582:	(
while_lstm_cell_15_604584:	,
while_lstm_cell_15_604586:	 Ђ*while/lstm_cell_15/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_604582_0while_lstm_cell_15_604584_0while_lstm_cell_15_604586_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6045442,
*while/lstm_cell_15/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_15/StatefulPartitionedCall*"
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
while_lstm_cell_15_604582while_lstm_cell_15_604582_0"8
while_lstm_cell_15_604584while_lstm_cell_15_604584_0"8
while_lstm_cell_15_604586while_lstm_cell_15_604586_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
ЬB
т
__inference__traced_save_608324
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_15_lstm_cell_15_kernel_read_readvariableopD
@savev2_lstm_15_lstm_cell_15_recurrent_kernel_read_readvariableop8
4savev2_lstm_15_lstm_cell_15_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableopA
=savev2_adam_lstm_15_lstm_cell_15_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_15_lstm_cell_15_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_15_lstm_cell_15_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableopA
=savev2_adam_lstm_15_lstm_cell_15_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_15_lstm_cell_15_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_15_lstm_cell_15_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_15_lstm_cell_15_kernel_read_readvariableop@savev2_lstm_15_lstm_cell_15_recurrent_kernel_read_readvariableop4savev2_lstm_15_lstm_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop=savev2_adam_lstm_15_lstm_cell_15_kernel_m_read_readvariableopGsavev2_adam_lstm_15_lstm_cell_15_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_15_lstm_cell_15_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop=savev2_adam_lstm_15_lstm_cell_15_kernel_v_read_readvariableopGsavev2_adam_lstm_15_lstm_cell_15_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_15_lstm_cell_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
л+
Ћ
H__inference_sequential_6_layer_call_and_return_conditional_losses_606059
input_7!
lstm_15_606028:	
lstm_15_606030:	!
lstm_15_606032:	 !
dense_18_606035:  
dense_18_606037: !
dense_19_606040: 
dense_19_606042:
identityЂ dense_18/StatefulPartitionedCallЂ dense_19/StatefulPartitionedCallЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂlstm_15/StatefulPartitionedCallЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ
lstm_15/StatefulPartitionedCallStatefulPartitionedCallinput_7lstm_15_606028lstm_15_606030lstm_15_606032*
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6058912!
lstm_15/StatefulPartitionedCallЖ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_18_606035dense_18_606037*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_6054722"
 dense_18/StatefulPartitionedCallЗ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_606040dense_19_606042*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_6054942"
 dense_19/StatefulPartitionedCallў
reshape_9/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_6055132
reshape_9/PartitionedCallЮ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_15_606028*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЎ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_606042*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mul
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall0^dense_19/bias/Regularizer/Square/ReadVariableOp ^lstm_15/StatefulPartitionedCall>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
Ї
Є	
while_body_605320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Ѕ
while/lstm_cell_15/mulMulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЉ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Љ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Љ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
оЁ
Ї
C__inference_lstm_15_layer_call_and_return_conditional_losses_607541

inputs=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileD
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_607408*
condR
while_cond_607407*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ
М
lstm_15_while_body_606208,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0:	K
<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0:	G
4lstm_15_while_lstm_cell_15_readvariableop_resource_0:	 
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorK
8lstm_15_while_lstm_cell_15_split_readvariableop_resource:	I
:lstm_15_while_lstm_cell_15_split_1_readvariableop_resource:	E
2lstm_15_while_lstm_cell_15_readvariableop_resource:	 Ђ)lstm_15/while/lstm_cell_15/ReadVariableOpЂ+lstm_15/while/lstm_cell_15/ReadVariableOp_1Ђ+lstm_15/while/lstm_cell_15/ReadVariableOp_2Ђ+lstm_15/while/lstm_cell_15/ReadVariableOp_3Ђ/lstm_15/while/lstm_cell_15/split/ReadVariableOpЂ1lstm_15/while/lstm_cell_15/split_1/ReadVariableOpг
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_15/while/lstm_cell_15/ones_like/ShapeShapelstm_15_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_15/while/lstm_cell_15/ones_like/Shape
*lstm_15/while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_15/while/lstm_cell_15/ones_like/Const№
$lstm_15/while/lstm_cell_15/ones_likeFill3lstm_15/while/lstm_cell_15/ones_like/Shape:output:03lstm_15/while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/ones_like
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dimо
/lstm_15/while/lstm_cell_15/split/ReadVariableOpReadVariableOp:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_15/while/lstm_cell_15/split/ReadVariableOp
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:07lstm_15/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_15/while/lstm_cell_15/splitч
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_15/while/lstm_cell_15/MatMulы
#lstm_15/while/lstm_cell_15/MatMul_1MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_1ы
#lstm_15/while/lstm_cell_15/MatMul_2MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_2ы
#lstm_15/while/lstm_cell_15/MatMul_3MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_3
,lstm_15/while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_15/while/lstm_cell_15/split_1/split_dimр
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp
"lstm_15/while/lstm_cell_15/split_1Split5lstm_15/while/lstm_cell_15/split_1/split_dim:output:09lstm_15/while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_15/while/lstm_cell_15/split_1п
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd+lstm_15/while/lstm_cell_15/MatMul:product:0+lstm_15/while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/while/lstm_cell_15/BiasAddх
$lstm_15/while/lstm_cell_15/BiasAdd_1BiasAdd-lstm_15/while/lstm_cell_15/MatMul_1:product:0+lstm_15/while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_1х
$lstm_15/while/lstm_cell_15/BiasAdd_2BiasAdd-lstm_15/while/lstm_cell_15/MatMul_2:product:0+lstm_15/while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_2х
$lstm_15/while/lstm_cell_15/BiasAdd_3BiasAdd-lstm_15/while/lstm_cell_15/MatMul_3:product:0+lstm_15/while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_3Х
lstm_15/while/lstm_cell_15/mulMullstm_15_while_placeholder_2-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/while/lstm_cell_15/mulЩ
 lstm_15/while/lstm_cell_15/mul_1Mullstm_15_while_placeholder_2-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_1Щ
 lstm_15/while/lstm_cell_15/mul_2Mullstm_15_while_placeholder_2-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_2Щ
 lstm_15/while/lstm_cell_15/mul_3Mullstm_15_while_placeholder_2-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_3Ь
)lstm_15/while/lstm_cell_15/ReadVariableOpReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_15/while/lstm_cell_15/ReadVariableOpБ
.lstm_15/while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_15/while/lstm_cell_15/strided_slice/stackЕ
0lstm_15/while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_15/while/lstm_cell_15/strided_slice/stack_1Е
0lstm_15/while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_15/while/lstm_cell_15/strided_slice/stack_2
(lstm_15/while/lstm_cell_15/strided_sliceStridedSlice1lstm_15/while/lstm_cell_15/ReadVariableOp:value:07lstm_15/while/lstm_cell_15/strided_slice/stack:output:09lstm_15/while/lstm_cell_15/strided_slice/stack_1:output:09lstm_15/while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_15/while/lstm_cell_15/strided_sliceн
#lstm_15/while/lstm_cell_15/MatMul_4MatMul"lstm_15/while/lstm_cell_15/mul:z:01lstm_15/while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_4з
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/BiasAdd:output:0-lstm_15/while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/while/lstm_cell_15/addЉ
"lstm_15/while/lstm_cell_15/SigmoidSigmoid"lstm_15/while/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/while/lstm_cell_15/Sigmoidа
+lstm_15/while/lstm_cell_15/ReadVariableOp_1ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_1Е
0lstm_15/while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_15/while/lstm_cell_15/strided_slice_1/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_1StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_1:value:09lstm_15/while/lstm_cell_15/strided_slice_1/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_1/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_1с
#lstm_15/while/lstm_cell_15/MatMul_5MatMul$lstm_15/while/lstm_cell_15/mul_1:z:03lstm_15/while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_5н
 lstm_15/while/lstm_cell_15/add_1AddV2-lstm_15/while/lstm_cell_15/BiasAdd_1:output:0-lstm_15/while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_1Џ
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/Sigmoid_1Ф
 lstm_15/while/lstm_cell_15/mul_4Mul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_4а
+lstm_15/while/lstm_cell_15/ReadVariableOp_2ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_2Е
0lstm_15/while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_15/while/lstm_cell_15/strided_slice_2/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_2StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_2:value:09lstm_15/while/lstm_cell_15/strided_slice_2/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_2/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_2с
#lstm_15/while/lstm_cell_15/MatMul_6MatMul$lstm_15/while/lstm_cell_15/mul_2:z:03lstm_15/while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_6н
 lstm_15/while/lstm_cell_15/add_2AddV2-lstm_15/while/lstm_cell_15/BiasAdd_2:output:0-lstm_15/while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_2Ђ
lstm_15/while/lstm_cell_15/ReluRelu$lstm_15/while/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_15/while/lstm_cell_15/Reluд
 lstm_15/while/lstm_cell_15/mul_5Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_5Ы
 lstm_15/while/lstm_cell_15/add_3AddV2$lstm_15/while/lstm_cell_15/mul_4:z:0$lstm_15/while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_3а
+lstm_15/while/lstm_cell_15/ReadVariableOp_3ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_3Е
0lstm_15/while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_15/while/lstm_cell_15/strided_slice_3/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_3StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_3:value:09lstm_15/while/lstm_cell_15/strided_slice_3/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_3/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_3с
#lstm_15/while/lstm_cell_15/MatMul_7MatMul$lstm_15/while/lstm_cell_15/mul_3:z:03lstm_15/while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_7н
 lstm_15/while/lstm_cell_15/add_4AddV2-lstm_15/while/lstm_cell_15/BiasAdd_3:output:0-lstm_15/while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_4Џ
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid$lstm_15/while/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/Sigmoid_2І
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_15/while/lstm_cell_15/Relu_1и
 lstm_15/while/lstm_cell_15/mul_6Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_6
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1lstm_15_while_placeholder$lstm_15/while/lstm_cell_15/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_15/while/TensorArrayV2Write/TensorListSetIteml
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add/y
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/addp
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add_1/y
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/IdentityІ
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2К
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3­
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_6:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/while/Identity_4­
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_3:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/while/Identity_5
lstm_15/while/NoOpNoOp*^lstm_15/while/lstm_cell_15/ReadVariableOp,^lstm_15/while/lstm_cell_15/ReadVariableOp_1,^lstm_15/while/lstm_cell_15/ReadVariableOp_2,^lstm_15/while/lstm_cell_15/ReadVariableOp_30^lstm_15/while/lstm_cell_15/split/ReadVariableOp2^lstm_15/while/lstm_cell_15/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_15/while/NoOp"9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"j
2lstm_15_while_lstm_cell_15_readvariableop_resource4lstm_15_while_lstm_cell_15_readvariableop_resource_0"z
:lstm_15_while_lstm_cell_15_split_1_readvariableop_resource<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0"v
8lstm_15_while_lstm_cell_15_split_readvariableop_resource:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0"Ш
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_15/while/lstm_cell_15/ReadVariableOp)lstm_15/while/lstm_cell_15/ReadVariableOp2Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_1+lstm_15/while/lstm_cell_15/ReadVariableOp_12Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_2+lstm_15/while/lstm_cell_15/ReadVariableOp_22Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_3+lstm_15/while/lstm_cell_15/ReadVariableOp_32b
/lstm_15/while/lstm_cell_15/split/ReadVariableOp/lstm_15/while/lstm_cell_15/split/ReadVariableOp2f
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp: 
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
while_cond_605319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_605319___redundant_placeholder04
0while_while_cond_605319___redundant_placeholder14
0while_while_cond_605319___redundant_placeholder24
0while_while_cond_605319___redundant_placeholder3
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
щ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_604777

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
seed2т 2&
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
seed2МЮђ2(
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
seed2Ќ2(
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
seed2ті2(
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2,
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
№Э
М
lstm_15_while_body_606511,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0:	K
<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0:	G
4lstm_15_while_lstm_cell_15_readvariableop_resource_0:	 
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorK
8lstm_15_while_lstm_cell_15_split_readvariableop_resource:	I
:lstm_15_while_lstm_cell_15_split_1_readvariableop_resource:	E
2lstm_15_while_lstm_cell_15_readvariableop_resource:	 Ђ)lstm_15/while/lstm_cell_15/ReadVariableOpЂ+lstm_15/while/lstm_cell_15/ReadVariableOp_1Ђ+lstm_15/while/lstm_cell_15/ReadVariableOp_2Ђ+lstm_15/while/lstm_cell_15/ReadVariableOp_3Ђ/lstm_15/while/lstm_cell_15/split/ReadVariableOpЂ1lstm_15/while/lstm_cell_15/split_1/ReadVariableOpг
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemЃ
*lstm_15/while/lstm_cell_15/ones_like/ShapeShapelstm_15_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_15/while/lstm_cell_15/ones_like/Shape
*lstm_15/while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*lstm_15/while/lstm_cell_15/ones_like/Const№
$lstm_15/while/lstm_cell_15/ones_likeFill3lstm_15/while/lstm_cell_15/ones_like/Shape:output:03lstm_15/while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/ones_like
(lstm_15/while/lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(lstm_15/while/lstm_cell_15/dropout/Constы
&lstm_15/while/lstm_cell_15/dropout/MulMul-lstm_15/while/lstm_cell_15/ones_like:output:01lstm_15/while/lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&lstm_15/while/lstm_cell_15/dropout/MulБ
(lstm_15/while/lstm_cell_15/dropout/ShapeShape-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_15/while/lstm_cell_15/dropout/ShapeЂ
?lstm_15/while/lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform1lstm_15/while/lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2нча2A
?lstm_15/while/lstm_cell_15/dropout/random_uniform/RandomUniformЋ
1lstm_15/while/lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1lstm_15/while/lstm_cell_15/dropout/GreaterEqual/yЊ
/lstm_15/while/lstm_cell_15/dropout/GreaterEqualGreaterEqualHlstm_15/while/lstm_cell_15/dropout/random_uniform/RandomUniform:output:0:lstm_15/while/lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/lstm_15/while/lstm_cell_15/dropout/GreaterEqualа
'lstm_15/while/lstm_cell_15/dropout/CastCast3lstm_15/while/lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_15/while/lstm_cell_15/dropout/Castц
(lstm_15/while/lstm_cell_15/dropout/Mul_1Mul*lstm_15/while/lstm_cell_15/dropout/Mul:z:0+lstm_15/while/lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_15/while/lstm_cell_15/dropout/Mul_1
*lstm_15/while/lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_15/while/lstm_cell_15/dropout_1/Constё
(lstm_15/while/lstm_cell_15/dropout_1/MulMul-lstm_15/while/lstm_cell_15/ones_like:output:03lstm_15/while/lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_15/while/lstm_cell_15/dropout_1/MulЕ
*lstm_15/while/lstm_cell_15/dropout_1/ShapeShape-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_15/while/lstm_cell_15/dropout_1/ShapeЈ
Alstm_15/while/lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_15/while/lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2фы2C
Alstm_15/while/lstm_cell_15/dropout_1/random_uniform/RandomUniformЏ
3lstm_15/while/lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_15/while/lstm_cell_15/dropout_1/GreaterEqual/yВ
1lstm_15/while/lstm_cell_15/dropout_1/GreaterEqualGreaterEqualJlstm_15/while/lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:0<lstm_15/while/lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_15/while/lstm_cell_15/dropout_1/GreaterEqualж
)lstm_15/while/lstm_cell_15/dropout_1/CastCast5lstm_15/while/lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_15/while/lstm_cell_15/dropout_1/Castю
*lstm_15/while/lstm_cell_15/dropout_1/Mul_1Mul,lstm_15/while/lstm_cell_15/dropout_1/Mul:z:0-lstm_15/while/lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_15/while/lstm_cell_15/dropout_1/Mul_1
*lstm_15/while/lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_15/while/lstm_cell_15/dropout_2/Constё
(lstm_15/while/lstm_cell_15/dropout_2/MulMul-lstm_15/while/lstm_cell_15/ones_like:output:03lstm_15/while/lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_15/while/lstm_cell_15/dropout_2/MulЕ
*lstm_15/while/lstm_cell_15/dropout_2/ShapeShape-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_15/while/lstm_cell_15/dropout_2/ShapeЈ
Alstm_15/while/lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_15/while/lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ю2C
Alstm_15/while/lstm_cell_15/dropout_2/random_uniform/RandomUniformЏ
3lstm_15/while/lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_15/while/lstm_cell_15/dropout_2/GreaterEqual/yВ
1lstm_15/while/lstm_cell_15/dropout_2/GreaterEqualGreaterEqualJlstm_15/while/lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:0<lstm_15/while/lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_15/while/lstm_cell_15/dropout_2/GreaterEqualж
)lstm_15/while/lstm_cell_15/dropout_2/CastCast5lstm_15/while/lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_15/while/lstm_cell_15/dropout_2/Castю
*lstm_15/while/lstm_cell_15/dropout_2/Mul_1Mul,lstm_15/while/lstm_cell_15/dropout_2/Mul:z:0-lstm_15/while/lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_15/while/lstm_cell_15/dropout_2/Mul_1
*lstm_15/while/lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2,
*lstm_15/while/lstm_cell_15/dropout_3/Constё
(lstm_15/while/lstm_cell_15/dropout_3/MulMul-lstm_15/while/lstm_cell_15/ones_like:output:03lstm_15/while/lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_15/while/lstm_cell_15/dropout_3/MulЕ
*lstm_15/while/lstm_cell_15/dropout_3/ShapeShape-lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2,
*lstm_15/while/lstm_cell_15/dropout_3/ShapeЈ
Alstm_15/while/lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_15/while/lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЭЋё2C
Alstm_15/while/lstm_cell_15/dropout_3/random_uniform/RandomUniformЏ
3lstm_15/while/lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>25
3lstm_15/while/lstm_cell_15/dropout_3/GreaterEqual/yВ
1lstm_15/while/lstm_cell_15/dropout_3/GreaterEqualGreaterEqualJlstm_15/while/lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:0<lstm_15/while/lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1lstm_15/while/lstm_cell_15/dropout_3/GreaterEqualж
)lstm_15/while/lstm_cell_15/dropout_3/CastCast5lstm_15/while/lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_15/while/lstm_cell_15/dropout_3/Castю
*lstm_15/while/lstm_cell_15/dropout_3/Mul_1Mul,lstm_15/while/lstm_cell_15/dropout_3/Mul:z:0-lstm_15/while/lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_15/while/lstm_cell_15/dropout_3/Mul_1
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dimо
/lstm_15/while/lstm_cell_15/split/ReadVariableOpReadVariableOp:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/lstm_15/while/lstm_cell_15/split/ReadVariableOp
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:07lstm_15/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2"
 lstm_15/while/lstm_cell_15/splitч
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_15/while/lstm_cell_15/MatMulы
#lstm_15/while/lstm_cell_15/MatMul_1MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_1ы
#lstm_15/while/lstm_cell_15/MatMul_2MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_2ы
#lstm_15/while/lstm_cell_15/MatMul_3MatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_3
,lstm_15/while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,lstm_15/while/lstm_cell_15/split_1/split_dimр
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp
"lstm_15/while/lstm_cell_15/split_1Split5lstm_15/while/lstm_cell_15/split_1/split_dim:output:09lstm_15/while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2$
"lstm_15/while/lstm_cell_15/split_1п
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd+lstm_15/while/lstm_cell_15/MatMul:product:0+lstm_15/while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/while/lstm_cell_15/BiasAddх
$lstm_15/while/lstm_cell_15/BiasAdd_1BiasAdd-lstm_15/while/lstm_cell_15/MatMul_1:product:0+lstm_15/while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_1х
$lstm_15/while/lstm_cell_15/BiasAdd_2BiasAdd-lstm_15/while/lstm_cell_15/MatMul_2:product:0+lstm_15/while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_2х
$lstm_15/while/lstm_cell_15/BiasAdd_3BiasAdd-lstm_15/while/lstm_cell_15/MatMul_3:product:0+lstm_15/while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/BiasAdd_3Ф
lstm_15/while/lstm_cell_15/mulMullstm_15_while_placeholder_2,lstm_15/while/lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/while/lstm_cell_15/mulЪ
 lstm_15/while/lstm_cell_15/mul_1Mullstm_15_while_placeholder_2.lstm_15/while/lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_1Ъ
 lstm_15/while/lstm_cell_15/mul_2Mullstm_15_while_placeholder_2.lstm_15/while/lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_2Ъ
 lstm_15/while/lstm_cell_15/mul_3Mullstm_15_while_placeholder_2.lstm_15/while/lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_3Ь
)lstm_15/while/lstm_cell_15/ReadVariableOpReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02+
)lstm_15/while/lstm_cell_15/ReadVariableOpБ
.lstm_15/while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_15/while/lstm_cell_15/strided_slice/stackЕ
0lstm_15/while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_15/while/lstm_cell_15/strided_slice/stack_1Е
0lstm_15/while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_15/while/lstm_cell_15/strided_slice/stack_2
(lstm_15/while/lstm_cell_15/strided_sliceStridedSlice1lstm_15/while/lstm_cell_15/ReadVariableOp:value:07lstm_15/while/lstm_cell_15/strided_slice/stack:output:09lstm_15/while/lstm_cell_15/strided_slice/stack_1:output:09lstm_15/while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_15/while/lstm_cell_15/strided_sliceн
#lstm_15/while/lstm_cell_15/MatMul_4MatMul"lstm_15/while/lstm_cell_15/mul:z:01lstm_15/while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_4з
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/BiasAdd:output:0-lstm_15/while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/while/lstm_cell_15/addЉ
"lstm_15/while/lstm_cell_15/SigmoidSigmoid"lstm_15/while/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/while/lstm_cell_15/Sigmoidа
+lstm_15/while/lstm_cell_15/ReadVariableOp_1ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_1Е
0lstm_15/while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_15/while/lstm_cell_15/strided_slice_1/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   24
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_1/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_1StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_1:value:09lstm_15/while/lstm_cell_15/strided_slice_1/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_1/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_1с
#lstm_15/while/lstm_cell_15/MatMul_5MatMul$lstm_15/while/lstm_cell_15/mul_1:z:03lstm_15/while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_5н
 lstm_15/while/lstm_cell_15/add_1AddV2-lstm_15/while/lstm_cell_15/BiasAdd_1:output:0-lstm_15/while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_1Џ
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/Sigmoid_1Ф
 lstm_15/while/lstm_cell_15/mul_4Mul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_4а
+lstm_15/while/lstm_cell_15/ReadVariableOp_2ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_2Е
0lstm_15/while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_15/while/lstm_cell_15/strided_slice_2/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   24
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_2/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_2StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_2:value:09lstm_15/while/lstm_cell_15/strided_slice_2/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_2/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_2с
#lstm_15/while/lstm_cell_15/MatMul_6MatMul$lstm_15/while/lstm_cell_15/mul_2:z:03lstm_15/while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_6н
 lstm_15/while/lstm_cell_15/add_2AddV2-lstm_15/while/lstm_cell_15/BiasAdd_2:output:0-lstm_15/while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_2Ђ
lstm_15/while/lstm_cell_15/ReluRelu$lstm_15/while/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_15/while/lstm_cell_15/Reluд
 lstm_15/while/lstm_cell_15/mul_5Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_5Ы
 lstm_15/while/lstm_cell_15/add_3AddV2$lstm_15/while/lstm_cell_15/mul_4:z:0$lstm_15/while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_3а
+lstm_15/while/lstm_cell_15/ReadVariableOp_3ReadVariableOp4lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02-
+lstm_15/while/lstm_cell_15/ReadVariableOp_3Е
0lstm_15/while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_15/while/lstm_cell_15/strided_slice_3/stackЙ
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_1Й
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2lstm_15/while/lstm_cell_15/strided_slice_3/stack_2Њ
*lstm_15/while/lstm_cell_15/strided_slice_3StridedSlice3lstm_15/while/lstm_cell_15/ReadVariableOp_3:value:09lstm_15/while/lstm_cell_15/strided_slice_3/stack:output:0;lstm_15/while/lstm_cell_15/strided_slice_3/stack_1:output:0;lstm_15/while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2,
*lstm_15/while/lstm_cell_15/strided_slice_3с
#lstm_15/while/lstm_cell_15/MatMul_7MatMul$lstm_15/while/lstm_cell_15/mul_3:z:03lstm_15/while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/while/lstm_cell_15/MatMul_7н
 lstm_15/while/lstm_cell_15/add_4AddV2-lstm_15/while/lstm_cell_15/BiasAdd_3:output:0-lstm_15/while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/add_4Џ
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid$lstm_15/while/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/while/lstm_cell_15/Sigmoid_2І
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_15/while/lstm_cell_15/Relu_1и
 lstm_15/while/lstm_cell_15/mul_6Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/while/lstm_cell_15/mul_6
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1lstm_15_while_placeholder$lstm_15/while/lstm_cell_15/mul_6:z:0*
_output_shapes
: *
element_dtype024
2lstm_15/while/TensorArrayV2Write/TensorListSetIteml
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add/y
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/addp
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add_1/y
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/IdentityІ
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2К
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3­
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_6:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/while/Identity_4­
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_3:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/while/Identity_5
lstm_15/while/NoOpNoOp*^lstm_15/while/lstm_cell_15/ReadVariableOp,^lstm_15/while/lstm_cell_15/ReadVariableOp_1,^lstm_15/while/lstm_cell_15/ReadVariableOp_2,^lstm_15/while/lstm_cell_15/ReadVariableOp_30^lstm_15/while/lstm_cell_15/split/ReadVariableOp2^lstm_15/while/lstm_cell_15/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_15/while/NoOp"9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"j
2lstm_15_while_lstm_cell_15_readvariableop_resource4lstm_15_while_lstm_cell_15_readvariableop_resource_0"z
:lstm_15_while_lstm_cell_15_split_1_readvariableop_resource<lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0"v
8lstm_15_while_lstm_cell_15_split_readvariableop_resource:lstm_15_while_lstm_cell_15_split_readvariableop_resource_0"Ш
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2V
)lstm_15/while/lstm_cell_15/ReadVariableOp)lstm_15/while/lstm_cell_15/ReadVariableOp2Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_1+lstm_15/while/lstm_cell_15/ReadVariableOp_12Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_2+lstm_15/while/lstm_cell_15/ReadVariableOp_22Z
+lstm_15/while/lstm_cell_15/ReadVariableOp_3+lstm_15/while/lstm_cell_15/ReadVariableOp_32b
/lstm_15/while/lstm_cell_15/split/ReadVariableOp/lstm_15/while/lstm_cell_15/split/ReadVariableOp2f
1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp1lstm_15/while/lstm_cell_15/split_1/ReadVariableOp: 
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
Э
ч
&sequential_6_lstm_15_while_cond_604270F
Bsequential_6_lstm_15_while_sequential_6_lstm_15_while_loop_counterL
Hsequential_6_lstm_15_while_sequential_6_lstm_15_while_maximum_iterations*
&sequential_6_lstm_15_while_placeholder,
(sequential_6_lstm_15_while_placeholder_1,
(sequential_6_lstm_15_while_placeholder_2,
(sequential_6_lstm_15_while_placeholder_3H
Dsequential_6_lstm_15_while_less_sequential_6_lstm_15_strided_slice_1^
Zsequential_6_lstm_15_while_sequential_6_lstm_15_while_cond_604270___redundant_placeholder0^
Zsequential_6_lstm_15_while_sequential_6_lstm_15_while_cond_604270___redundant_placeholder1^
Zsequential_6_lstm_15_while_sequential_6_lstm_15_while_cond_604270___redundant_placeholder2^
Zsequential_6_lstm_15_while_sequential_6_lstm_15_while_cond_604270___redundant_placeholder3'
#sequential_6_lstm_15_while_identity
й
sequential_6/lstm_15/while/LessLess&sequential_6_lstm_15_while_placeholderDsequential_6_lstm_15_while_less_sequential_6_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_15/while/Less
#sequential_6/lstm_15/while/IdentityIdentity#sequential_6/lstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_15/while/Identity"S
#sequential_6_lstm_15_while_identity,sequential_6/lstm_15/while/Identity:output:0*(
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
О
З
(__inference_lstm_15_layer_call_fn_607859
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6046332
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
р	
І
-__inference_sequential_6_layer_call_fn_606723

inputs
unknown:	
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
H__inference_sequential_6_layer_call_and_return_conditional_losses_6055282
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
ЦЖ

&sequential_6_lstm_15_while_body_604271F
Bsequential_6_lstm_15_while_sequential_6_lstm_15_while_loop_counterL
Hsequential_6_lstm_15_while_sequential_6_lstm_15_while_maximum_iterations*
&sequential_6_lstm_15_while_placeholder,
(sequential_6_lstm_15_while_placeholder_1,
(sequential_6_lstm_15_while_placeholder_2,
(sequential_6_lstm_15_while_placeholder_3E
Asequential_6_lstm_15_while_sequential_6_lstm_15_strided_slice_1_0
}sequential_6_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_15_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_6_lstm_15_while_lstm_cell_15_split_readvariableop_resource_0:	X
Isequential_6_lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0:	T
Asequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0:	 '
#sequential_6_lstm_15_while_identity)
%sequential_6_lstm_15_while_identity_1)
%sequential_6_lstm_15_while_identity_2)
%sequential_6_lstm_15_while_identity_3)
%sequential_6_lstm_15_while_identity_4)
%sequential_6_lstm_15_while_identity_5C
?sequential_6_lstm_15_while_sequential_6_lstm_15_strided_slice_1
{sequential_6_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_15_tensorarrayunstack_tensorlistfromtensorX
Esequential_6_lstm_15_while_lstm_cell_15_split_readvariableop_resource:	V
Gsequential_6_lstm_15_while_lstm_cell_15_split_1_readvariableop_resource:	R
?sequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource:	 Ђ6sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOpЂ8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_1Ђ8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_2Ђ8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_3Ђ<sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOpЂ>sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOpэ
Lsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2N
Lsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeб
>sequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_15_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_15_while_placeholderUsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02@
>sequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItemЪ
7sequential_6/lstm_15/while/lstm_cell_15/ones_like/ShapeShape(sequential_6_lstm_15_while_placeholder_2*
T0*
_output_shapes
:29
7sequential_6/lstm_15/while/lstm_cell_15/ones_like/ShapeЗ
7sequential_6/lstm_15/while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7sequential_6/lstm_15/while/lstm_cell_15/ones_like/ConstЄ
1sequential_6/lstm_15/while/lstm_cell_15/ones_likeFill@sequential_6/lstm_15/while/lstm_cell_15/ones_like/Shape:output:0@sequential_6/lstm_15/while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/ones_likeД
7sequential_6/lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_15/while/lstm_cell_15/split/split_dim
<sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOpReadVariableOpGsequential_6_lstm_15_while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOpЧ
-sequential_6/lstm_15/while/lstm_cell_15/splitSplit@sequential_6/lstm_15/while/lstm_cell_15/split/split_dim:output:0Dsequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2/
-sequential_6/lstm_15/while/lstm_cell_15/split
.sequential_6/lstm_15/while/lstm_cell_15/MatMulMatMulEsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_6/lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_6/lstm_15/while/lstm_cell_15/MatMul
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_1MatMulEsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_6/lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_1
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_2MatMulEsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_6/lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_2
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_3MatMulEsequential_6/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:06sequential_6/lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_3И
9sequential_6/lstm_15/while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9sequential_6/lstm_15/while/lstm_cell_15/split_1/split_dim
>sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOpReadVariableOpIsequential_6_lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOpП
/sequential_6/lstm_15/while/lstm_cell_15/split_1SplitBsequential_6/lstm_15/while/lstm_cell_15/split_1/split_dim:output:0Fsequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split21
/sequential_6/lstm_15/while/lstm_cell_15/split_1
/sequential_6/lstm_15/while/lstm_cell_15/BiasAddBiasAdd8sequential_6/lstm_15/while/lstm_cell_15/MatMul:product:08sequential_6/lstm_15/while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_6/lstm_15/while/lstm_cell_15/BiasAdd
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_1BiasAdd:sequential_6/lstm_15/while/lstm_cell_15/MatMul_1:product:08sequential_6/lstm_15/while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_1
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_2BiasAdd:sequential_6/lstm_15/while/lstm_cell_15/MatMul_2:product:08sequential_6/lstm_15/while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_2
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_3BiasAdd:sequential_6/lstm_15/while/lstm_cell_15/MatMul_3:product:08sequential_6/lstm_15/while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_3љ
+sequential_6/lstm_15/while/lstm_cell_15/mulMul(sequential_6_lstm_15_while_placeholder_2:sequential_6/lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/while/lstm_cell_15/mul§
-sequential_6/lstm_15/while/lstm_cell_15/mul_1Mul(sequential_6_lstm_15_while_placeholder_2:sequential_6/lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_1§
-sequential_6/lstm_15/while/lstm_cell_15/mul_2Mul(sequential_6_lstm_15_while_placeholder_2:sequential_6/lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_2§
-sequential_6/lstm_15/while/lstm_cell_15/mul_3Mul(sequential_6_lstm_15_while_placeholder_2:sequential_6/lstm_15/while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_3ѓ
6sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOpReadVariableOpAsequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOpЫ
;sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stackЯ
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_1Я
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_2ь
5sequential_6/lstm_15/while/lstm_cell_15/strided_sliceStridedSlice>sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp:value:0Dsequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack:output:0Fsequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_1:output:0Fsequential_6/lstm_15/while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_6/lstm_15/while/lstm_cell_15/strided_slice
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_4MatMul/sequential_6/lstm_15/while/lstm_cell_15/mul:z:0>sequential_6/lstm_15/while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_4
+sequential_6/lstm_15/while/lstm_cell_15/addAddV28sequential_6/lstm_15/while/lstm_cell_15/BiasAdd:output:0:sequential_6/lstm_15/while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/while/lstm_cell_15/addа
/sequential_6/lstm_15/while/lstm_cell_15/SigmoidSigmoid/sequential_6/lstm_15/while/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_6/lstm_15/while/lstm_cell_15/Sigmoidї
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_1ReadVariableOpAsequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02:
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_1Я
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stackг
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_1г
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_2ј
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1StridedSlice@sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_1:value:0Fsequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_1:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_5MatMul1sequential_6/lstm_15/while/lstm_cell_15/mul_1:z:0@sequential_6/lstm_15/while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_5
-sequential_6/lstm_15/while/lstm_cell_15/add_1AddV2:sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_1:output:0:sequential_6/lstm_15/while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/add_1ж
1sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid1sequential_6/lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_1ј
-sequential_6/lstm_15/while/lstm_cell_15/mul_4Mul5sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_1:y:0(sequential_6_lstm_15_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_4ї
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_2ReadVariableOpAsequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02:
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_2Я
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2?
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stackг
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_1г
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_2ј
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2StridedSlice@sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_2:value:0Fsequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_1:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_6MatMul1sequential_6/lstm_15/while/lstm_cell_15/mul_2:z:0@sequential_6/lstm_15/while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_6
-sequential_6/lstm_15/while/lstm_cell_15/add_2AddV2:sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_2:output:0:sequential_6/lstm_15/while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/add_2Щ
,sequential_6/lstm_15/while/lstm_cell_15/ReluRelu1sequential_6/lstm_15/while/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_6/lstm_15/while/lstm_cell_15/Relu
-sequential_6/lstm_15/while/lstm_cell_15/mul_5Mul3sequential_6/lstm_15/while/lstm_cell_15/Sigmoid:y:0:sequential_6/lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_5џ
-sequential_6/lstm_15/while/lstm_cell_15/add_3AddV21sequential_6/lstm_15/while/lstm_cell_15/mul_4:z:01sequential_6/lstm_15/while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/add_3ї
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_3ReadVariableOpAsequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02:
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_3Я
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2?
=sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stackг
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_1г
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_2ј
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3StridedSlice@sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_3:value:0Fsequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_1:output:0Hsequential_6/lstm_15/while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask29
7sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_7MatMul1sequential_6/lstm_15/while/lstm_cell_15/mul_3:z:0@sequential_6/lstm_15/while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_6/lstm_15/while/lstm_cell_15/MatMul_7
-sequential_6/lstm_15/while/lstm_cell_15/add_4AddV2:sequential_6/lstm_15/while/lstm_cell_15/BiasAdd_3:output:0:sequential_6/lstm_15/while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/add_4ж
1sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid1sequential_6/lstm_15/while/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 23
1sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_2Э
.sequential_6/lstm_15/while/lstm_cell_15/Relu_1Relu1sequential_6/lstm_15/while/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_6/lstm_15/while/lstm_cell_15/Relu_1
-sequential_6/lstm_15/while/lstm_cell_15/mul_6Mul5sequential_6/lstm_15/while/lstm_cell_15/Sigmoid_2:y:0<sequential_6/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_6/lstm_15/while/lstm_cell_15/mul_6Щ
?sequential_6/lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_15_while_placeholder_1&sequential_6_lstm_15_while_placeholder1sequential_6/lstm_15/while/lstm_cell_15/mul_6:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_15/while/TensorArrayV2Write/TensorListSetItem
 sequential_6/lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_15/while/add/yН
sequential_6/lstm_15/while/addAddV2&sequential_6_lstm_15_while_placeholder)sequential_6/lstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_15/while/add
"sequential_6/lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_15/while/add_1/yп
 sequential_6/lstm_15/while/add_1AddV2Bsequential_6_lstm_15_while_sequential_6_lstm_15_while_loop_counter+sequential_6/lstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_15/while/add_1П
#sequential_6/lstm_15/while/IdentityIdentity$sequential_6/lstm_15/while/add_1:z:0 ^sequential_6/lstm_15/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_15/while/Identityч
%sequential_6/lstm_15/while/Identity_1IdentityHsequential_6_lstm_15_while_sequential_6_lstm_15_while_maximum_iterations ^sequential_6/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_15/while/Identity_1С
%sequential_6/lstm_15/while/Identity_2Identity"sequential_6/lstm_15/while/add:z:0 ^sequential_6/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_15/while/Identity_2ю
%sequential_6/lstm_15/while/Identity_3IdentityOsequential_6/lstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_15/while/Identity_3с
%sequential_6/lstm_15/while/Identity_4Identity1sequential_6/lstm_15/while/lstm_cell_15/mul_6:z:0 ^sequential_6/lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_6/lstm_15/while/Identity_4с
%sequential_6/lstm_15/while/Identity_5Identity1sequential_6/lstm_15/while/lstm_cell_15/add_3:z:0 ^sequential_6/lstm_15/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_6/lstm_15/while/Identity_5ю
sequential_6/lstm_15/while/NoOpNoOp7^sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp9^sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_19^sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_29^sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_3=^sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOp?^sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_6/lstm_15/while/NoOp"S
#sequential_6_lstm_15_while_identity,sequential_6/lstm_15/while/Identity:output:0"W
%sequential_6_lstm_15_while_identity_1.sequential_6/lstm_15/while/Identity_1:output:0"W
%sequential_6_lstm_15_while_identity_2.sequential_6/lstm_15/while/Identity_2:output:0"W
%sequential_6_lstm_15_while_identity_3.sequential_6/lstm_15/while/Identity_3:output:0"W
%sequential_6_lstm_15_while_identity_4.sequential_6/lstm_15/while/Identity_4:output:0"W
%sequential_6_lstm_15_while_identity_5.sequential_6/lstm_15/while/Identity_5:output:0"
?sequential_6_lstm_15_while_lstm_cell_15_readvariableop_resourceAsequential_6_lstm_15_while_lstm_cell_15_readvariableop_resource_0"
Gsequential_6_lstm_15_while_lstm_cell_15_split_1_readvariableop_resourceIsequential_6_lstm_15_while_lstm_cell_15_split_1_readvariableop_resource_0"
Esequential_6_lstm_15_while_lstm_cell_15_split_readvariableop_resourceGsequential_6_lstm_15_while_lstm_cell_15_split_readvariableop_resource_0"
?sequential_6_lstm_15_while_sequential_6_lstm_15_strided_slice_1Asequential_6_lstm_15_while_sequential_6_lstm_15_strided_slice_1_0"ќ
{sequential_6_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_15_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2p
6sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp6sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp2t
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_18sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_12t
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_28sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_22t
8sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_38sequential_6/lstm_15/while/lstm_cell_15/ReadVariableOp_32|
<sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOp<sequential_6/lstm_15/while/lstm_cell_15/split/ReadVariableOp2
>sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOp>sequential_6/lstm_15/while/lstm_cell_15/split_1/ReadVariableOp: 
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


H__inference_sequential_6_layer_call_and_return_conditional_losses_606704

inputsE
2lstm_15_lstm_cell_15_split_readvariableop_resource:	C
4lstm_15_lstm_cell_15_split_1_readvariableop_resource:	?
,lstm_15_lstm_cell_15_readvariableop_resource:	 9
'dense_18_matmul_readvariableop_resource:  6
(dense_18_biasadd_readvariableop_resource: 9
'dense_19_matmul_readvariableop_resource: 6
(dense_19_biasadd_readvariableop_resource:
identityЂdense_18/BiasAdd/ReadVariableOpЂdense_18/MatMul/ReadVariableOpЂdense_19/BiasAdd/ReadVariableOpЂdense_19/MatMul/ReadVariableOpЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂ#lstm_15/lstm_cell_15/ReadVariableOpЂ%lstm_15/lstm_cell_15/ReadVariableOp_1Ђ%lstm_15/lstm_cell_15/ReadVariableOp_2Ђ%lstm_15/lstm_cell_15/ReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_15/lstm_cell_15/split/ReadVariableOpЂ+lstm_15/lstm_cell_15/split_1/ReadVariableOpЂlstm_15/whileT
lstm_15/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_15/Shape
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stack
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicel
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros/mul/y
lstm_15/zeros/mulMullstm_15/strided_slice:output:0lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/mulo
lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros/Less/y
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lessr
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros/packed/1Ѓ
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros/packedo
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros/Const
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/zerosp
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros_1/mul/y
lstm_15/zeros_1/mulMullstm_15/strided_slice:output:0lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/muls
lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros_1/Less/y
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessv
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros_1/packed/1Љ
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros_1/packeds
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros_1/Const
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/zeros_1
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/perm
lstm_15/transpose	Transposeinputslstm_15/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stack
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_15/TensorArrayV2/element_shapeв
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2Я
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensor
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stack
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2Ќ
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_15/strided_slice_2
$lstm_15/lstm_cell_15/ones_like/ShapeShapelstm_15/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_15/lstm_cell_15/ones_like/Shape
$lstm_15/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_15/lstm_cell_15/ones_like/Constи
lstm_15/lstm_cell_15/ones_likeFill-lstm_15/lstm_cell_15/ones_like/Shape:output:0-lstm_15/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/ones_like
"lstm_15/lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"lstm_15/lstm_cell_15/dropout/Constг
 lstm_15/lstm_cell_15/dropout/MulMul'lstm_15/lstm_cell_15/ones_like:output:0+lstm_15/lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_15/lstm_cell_15/dropout/Mul
"lstm_15/lstm_cell_15/dropout/ShapeShape'lstm_15/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_15/lstm_cell_15/dropout/Shape
9lstm_15/lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform+lstm_15/lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ТБE2;
9lstm_15/lstm_cell_15/dropout/random_uniform/RandomUniform
+lstm_15/lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+lstm_15/lstm_cell_15/dropout/GreaterEqual/y
)lstm_15/lstm_cell_15/dropout/GreaterEqualGreaterEqualBlstm_15/lstm_cell_15/dropout/random_uniform/RandomUniform:output:04lstm_15/lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_15/lstm_cell_15/dropout/GreaterEqualО
!lstm_15/lstm_cell_15/dropout/CastCast-lstm_15/lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_15/lstm_cell_15/dropout/CastЮ
"lstm_15/lstm_cell_15/dropout/Mul_1Mul$lstm_15/lstm_cell_15/dropout/Mul:z:0%lstm_15/lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/lstm_cell_15/dropout/Mul_1
$lstm_15/lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_15/lstm_cell_15/dropout_1/Constй
"lstm_15/lstm_cell_15/dropout_1/MulMul'lstm_15/lstm_cell_15/ones_like:output:0-lstm_15/lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/lstm_cell_15/dropout_1/MulЃ
$lstm_15/lstm_cell_15/dropout_1/ShapeShape'lstm_15/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_15/lstm_cell_15/dropout_1/Shape
;lstm_15/lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_15/lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ћ2=
;lstm_15/lstm_cell_15/dropout_1/random_uniform/RandomUniformЃ
-lstm_15/lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_15/lstm_cell_15/dropout_1/GreaterEqual/y
+lstm_15/lstm_cell_15/dropout_1/GreaterEqualGreaterEqualDlstm_15/lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:06lstm_15/lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_15/lstm_cell_15/dropout_1/GreaterEqualФ
#lstm_15/lstm_cell_15/dropout_1/CastCast/lstm_15/lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/lstm_cell_15/dropout_1/Castж
$lstm_15/lstm_cell_15/dropout_1/Mul_1Mul&lstm_15/lstm_cell_15/dropout_1/Mul:z:0'lstm_15/lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/lstm_cell_15/dropout_1/Mul_1
$lstm_15/lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_15/lstm_cell_15/dropout_2/Constй
"lstm_15/lstm_cell_15/dropout_2/MulMul'lstm_15/lstm_cell_15/ones_like:output:0-lstm_15/lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/lstm_cell_15/dropout_2/MulЃ
$lstm_15/lstm_cell_15/dropout_2/ShapeShape'lstm_15/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_15/lstm_cell_15/dropout_2/Shape
;lstm_15/lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_15/lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ѕ2=
;lstm_15/lstm_cell_15/dropout_2/random_uniform/RandomUniformЃ
-lstm_15/lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_15/lstm_cell_15/dropout_2/GreaterEqual/y
+lstm_15/lstm_cell_15/dropout_2/GreaterEqualGreaterEqualDlstm_15/lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:06lstm_15/lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_15/lstm_cell_15/dropout_2/GreaterEqualФ
#lstm_15/lstm_cell_15/dropout_2/CastCast/lstm_15/lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/lstm_cell_15/dropout_2/Castж
$lstm_15/lstm_cell_15/dropout_2/Mul_1Mul&lstm_15/lstm_cell_15/dropout_2/Mul:z:0'lstm_15/lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/lstm_cell_15/dropout_2/Mul_1
$lstm_15/lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2&
$lstm_15/lstm_cell_15/dropout_3/Constй
"lstm_15/lstm_cell_15/dropout_3/MulMul'lstm_15/lstm_cell_15/ones_like:output:0-lstm_15/lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_15/lstm_cell_15/dropout_3/MulЃ
$lstm_15/lstm_cell_15/dropout_3/ShapeShape'lstm_15/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_15/lstm_cell_15/dropout_3/Shape
;lstm_15/lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_15/lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЭN2=
;lstm_15/lstm_cell_15/dropout_3/random_uniform/RandomUniformЃ
-lstm_15/lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2/
-lstm_15/lstm_cell_15/dropout_3/GreaterEqual/y
+lstm_15/lstm_cell_15/dropout_3/GreaterEqualGreaterEqualDlstm_15/lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:06lstm_15/lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+lstm_15/lstm_cell_15/dropout_3/GreaterEqualФ
#lstm_15/lstm_cell_15/dropout_3/CastCast/lstm_15/lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_15/lstm_cell_15/dropout_3/Castж
$lstm_15/lstm_cell_15/dropout_3/Mul_1Mul&lstm_15/lstm_cell_15/dropout_3/Mul:z:0'lstm_15/lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$lstm_15/lstm_cell_15/dropout_3/Mul_1
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dimЪ
)lstm_15/lstm_cell_15/split/ReadVariableOpReadVariableOp2lstm_15_lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_15/lstm_cell_15/split/ReadVariableOpћ
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:01lstm_15/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_15/lstm_cell_15/splitН
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMulС
lstm_15/lstm_cell_15/MatMul_1MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_1С
lstm_15/lstm_cell_15/MatMul_2MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_2С
lstm_15/lstm_cell_15/MatMul_3MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_3
&lstm_15/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_15/lstm_cell_15/split_1/split_dimЬ
+lstm_15/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_15/lstm_cell_15/split_1/ReadVariableOpѓ
lstm_15/lstm_cell_15/split_1Split/lstm_15/lstm_cell_15/split_1/split_dim:output:03lstm_15/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_15/lstm_cell_15/split_1Ч
lstm_15/lstm_cell_15/BiasAddBiasAdd%lstm_15/lstm_cell_15/MatMul:product:0%lstm_15/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/BiasAddЭ
lstm_15/lstm_cell_15/BiasAdd_1BiasAdd'lstm_15/lstm_cell_15/MatMul_1:product:0%lstm_15/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_1Э
lstm_15/lstm_cell_15/BiasAdd_2BiasAdd'lstm_15/lstm_cell_15/MatMul_2:product:0%lstm_15/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_2Э
lstm_15/lstm_cell_15/BiasAdd_3BiasAdd'lstm_15/lstm_cell_15/MatMul_3:product:0%lstm_15/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_3­
lstm_15/lstm_cell_15/mulMullstm_15/zeros:output:0&lstm_15/lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mulГ
lstm_15/lstm_cell_15/mul_1Mullstm_15/zeros:output:0(lstm_15/lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_1Г
lstm_15/lstm_cell_15/mul_2Mullstm_15/zeros:output:0(lstm_15/lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_2Г
lstm_15/lstm_cell_15/mul_3Mullstm_15/zeros:output:0(lstm_15/lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_3И
#lstm_15/lstm_cell_15/ReadVariableOpReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_15/lstm_cell_15/ReadVariableOpЅ
(lstm_15/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_15/lstm_cell_15/strided_slice/stackЉ
*lstm_15/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_15/lstm_cell_15/strided_slice/stack_1Љ
*lstm_15/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_15/lstm_cell_15/strided_slice/stack_2њ
"lstm_15/lstm_cell_15/strided_sliceStridedSlice+lstm_15/lstm_cell_15/ReadVariableOp:value:01lstm_15/lstm_cell_15/strided_slice/stack:output:03lstm_15/lstm_cell_15/strided_slice/stack_1:output:03lstm_15/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_15/lstm_cell_15/strided_sliceХ
lstm_15/lstm_cell_15/MatMul_4MatMullstm_15/lstm_cell_15/mul:z:0+lstm_15/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_4П
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/BiasAdd:output:0'lstm_15/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add
lstm_15/lstm_cell_15/SigmoidSigmoidlstm_15/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/SigmoidМ
%lstm_15/lstm_cell_15/ReadVariableOp_1ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_1Љ
*lstm_15/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_15/lstm_cell_15/strided_slice_1/stack­
,lstm_15/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_15/lstm_cell_15/strided_slice_1/stack_1­
,lstm_15/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_1/stack_2
$lstm_15/lstm_cell_15/strided_slice_1StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_1:value:03lstm_15/lstm_cell_15/strided_slice_1/stack:output:05lstm_15/lstm_cell_15/strided_slice_1/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_1Щ
lstm_15/lstm_cell_15/MatMul_5MatMullstm_15/lstm_cell_15/mul_1:z:0-lstm_15/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_5Х
lstm_15/lstm_cell_15/add_1AddV2'lstm_15/lstm_cell_15/BiasAdd_1:output:0'lstm_15/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_1
lstm_15/lstm_cell_15/Sigmoid_1Sigmoidlstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/Sigmoid_1Џ
lstm_15/lstm_cell_15/mul_4Mul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_4М
%lstm_15/lstm_cell_15/ReadVariableOp_2ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_2Љ
*lstm_15/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_15/lstm_cell_15/strided_slice_2/stack­
,lstm_15/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_15/lstm_cell_15/strided_slice_2/stack_1­
,lstm_15/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_2/stack_2
$lstm_15/lstm_cell_15/strided_slice_2StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_2:value:03lstm_15/lstm_cell_15/strided_slice_2/stack:output:05lstm_15/lstm_cell_15/strided_slice_2/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_2Щ
lstm_15/lstm_cell_15/MatMul_6MatMullstm_15/lstm_cell_15/mul_2:z:0-lstm_15/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_6Х
lstm_15/lstm_cell_15/add_2AddV2'lstm_15/lstm_cell_15/BiasAdd_2:output:0'lstm_15/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_2
lstm_15/lstm_cell_15/ReluRelulstm_15/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/ReluМ
lstm_15/lstm_cell_15/mul_5Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_5Г
lstm_15/lstm_cell_15/add_3AddV2lstm_15/lstm_cell_15/mul_4:z:0lstm_15/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_3М
%lstm_15/lstm_cell_15/ReadVariableOp_3ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_3Љ
*lstm_15/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_15/lstm_cell_15/strided_slice_3/stack­
,lstm_15/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_15/lstm_cell_15/strided_slice_3/stack_1­
,lstm_15/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_3/stack_2
$lstm_15/lstm_cell_15/strided_slice_3StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_3:value:03lstm_15/lstm_cell_15/strided_slice_3/stack:output:05lstm_15/lstm_cell_15/strided_slice_3/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_3Щ
lstm_15/lstm_cell_15/MatMul_7MatMullstm_15/lstm_cell_15/mul_3:z:0-lstm_15/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_7Х
lstm_15/lstm_cell_15/add_4AddV2'lstm_15/lstm_cell_15/BiasAdd_3:output:0'lstm_15/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_4
lstm_15/lstm_cell_15/Sigmoid_2Sigmoidlstm_15/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/Sigmoid_2
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/Relu_1Р
lstm_15/lstm_cell_15/mul_6Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_6
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_15/TensorArrayV2_1/element_shapeи
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2_1^
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/time
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counterљ
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_15_lstm_cell_15_split_readvariableop_resource4lstm_15_lstm_cell_15_split_1_readvariableop_resource,lstm_15_lstm_cell_15_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_15_while_body_606511*%
condR
lstm_15_while_cond_606510*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_15/whileХ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStack
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_15/strided_slice_3/stack
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2Ъ
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_15/strided_slice_3
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/permХ
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_15/transpose_1v
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/runtimeЈ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_18/MatMul/ReadVariableOpЈ
dense_18/MatMulMatMul lstm_15/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/MatMulЇ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_18/BiasAdd/ReadVariableOpЅ
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/ReluЈ
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_19/MatMul/ReadVariableOpЃ
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19/MatMulЇ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOpЅ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19/BiasAddk
reshape_9/ShapeShapedense_19/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_9/Shape
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2в
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shapeЄ
reshape_9/ReshapeReshapedense_19/BiasAdd:output:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_9/Reshapeђ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_15_lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЧ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/muly
IdentityIdentityreshape_9/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp0^dense_19/bias/Regularizer/Square/ReadVariableOp$^lstm_15/lstm_cell_15/ReadVariableOp&^lstm_15/lstm_cell_15/ReadVariableOp_1&^lstm_15/lstm_cell_15/ReadVariableOp_2&^lstm_15/lstm_cell_15/ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*^lstm_15/lstm_cell_15/split/ReadVariableOp,^lstm_15/lstm_cell_15/split_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2J
#lstm_15/lstm_cell_15/ReadVariableOp#lstm_15/lstm_cell_15/ReadVariableOp2N
%lstm_15/lstm_cell_15/ReadVariableOp_1%lstm_15/lstm_cell_15/ReadVariableOp_12N
%lstm_15/lstm_cell_15/ReadVariableOp_2%lstm_15/lstm_cell_15/ReadVariableOp_22N
%lstm_15/lstm_cell_15/ReadVariableOp_3%lstm_15/lstm_cell_15/ReadVariableOp_32~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_15/lstm_cell_15/split/ReadVariableOp)lstm_15/lstm_cell_15/split/ReadVariableOp2Z
+lstm_15/lstm_cell_15/split_1/ReadVariableOp+lstm_15/lstm_cell_15/split_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
R
Х
C__inference_lstm_15_layer_call_and_return_conditional_losses_604633

inputs&
lstm_cell_15_604545:	"
lstm_cell_15_604547:	&
lstm_cell_15_604549:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_15/StatefulPartitionedCallЂwhileD
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
strided_slice_2
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_604545lstm_cell_15_604547lstm_cell_15_604549*
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6045442&
$lstm_cell_15/StatefulPartitionedCall
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
while/loop_counterР
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_604545lstm_cell_15_604547lstm_cell_15_604549*
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
while_body_604558*
condR
while_cond_604557*K
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
runtimeг
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_15_604545*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
у	
Ї
-__inference_sequential_6_layer_call_fn_605991
input_7
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
H__inference_sequential_6_layer_call_and_return_conditional_losses_6059552
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
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
зЯ
Ї
C__inference_lstm_15_layer_call_and_return_conditional_losses_605891

inputs=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileD
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like}
lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout/ConstГ
lstm_cell_15/dropout/MulMullstm_cell_15/ones_like:output:0#lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul
lstm_cell_15/dropout/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout/Shapeј
1lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2єХ23
1lstm_cell_15/dropout/random_uniform/RandomUniform
#lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_15/dropout/GreaterEqual/yђ
!lstm_cell_15/dropout/GreaterEqualGreaterEqual:lstm_cell_15/dropout/random_uniform/RandomUniform:output:0,lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_15/dropout/GreaterEqualІ
lstm_cell_15/dropout/CastCast%lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/CastЎ
lstm_cell_15/dropout/Mul_1Mullstm_cell_15/dropout/Mul:z:0lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul_1
lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_1/ConstЙ
lstm_cell_15/dropout_1/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul
lstm_cell_15/dropout_1/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_1/Shape§
3lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ъK25
3lstm_cell_15/dropout_1/random_uniform/RandomUniform
%lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_1/GreaterEqual/yњ
#lstm_cell_15/dropout_1/GreaterEqualGreaterEqual<lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_1/GreaterEqualЌ
lstm_cell_15/dropout_1/CastCast'lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/CastЖ
lstm_cell_15/dropout_1/Mul_1Mullstm_cell_15/dropout_1/Mul:z:0lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul_1
lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_2/ConstЙ
lstm_cell_15/dropout_2/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul
lstm_cell_15/dropout_2/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_2/Shape§
3lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2їЋ*25
3lstm_cell_15/dropout_2/random_uniform/RandomUniform
%lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_2/GreaterEqual/yњ
#lstm_cell_15/dropout_2/GreaterEqualGreaterEqual<lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_2/GreaterEqualЌ
lstm_cell_15/dropout_2/CastCast'lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/CastЖ
lstm_cell_15/dropout_2/Mul_1Mullstm_cell_15/dropout_2/Mul:z:0lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul_1
lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_3/ConstЙ
lstm_cell_15/dropout_3/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul
lstm_cell_15/dropout_3/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_3/Shapeў
3lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Ъ25
3lstm_cell_15/dropout_3/random_uniform/RandomUniform
%lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_3/GreaterEqual/yњ
#lstm_cell_15/dropout_3/GreaterEqualGreaterEqual<lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_3/GreaterEqualЌ
lstm_cell_15/dropout_3/CastCast'lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/CastЖ
lstm_cell_15/dropout_3/Mul_1Mullstm_cell_15/dropout_3/Mul:z:0lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul_1~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0 lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0 lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0 lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_605726*
condR
while_cond_605725*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
Е
(__inference_lstm_15_layer_call_fn_607881

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6054532
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
ё

)__inference_dense_19_layer_call_fn_607943

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
D__inference_dense_19_layer_call_and_return_conditional_losses_6054942
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
R
Х
C__inference_lstm_15_layer_call_and_return_conditional_losses_604930

inputs&
lstm_cell_15_604842:	"
lstm_cell_15_604844:	&
lstm_cell_15_604846:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_15/StatefulPartitionedCallЂwhileD
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
strided_slice_2
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_604842lstm_cell_15_604844lstm_cell_15_604846*
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6047772&
$lstm_cell_15/StatefulPartitionedCall
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
while/loop_counterР
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_604842lstm_cell_15_604844lstm_cell_15_604846*
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
while_body_604855*
condR
while_cond_604854*K
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
runtimeг
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_15_604842*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityН
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

a
E__inference_reshape_9_layer_call_and_return_conditional_losses_607956

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
и+
Њ
H__inference_sequential_6_layer_call_and_return_conditional_losses_605528

inputs!
lstm_15_605454:	
lstm_15_605456:	!
lstm_15_605458:	 !
dense_18_605473:  
dense_18_605475: !
dense_19_605495: 
dense_19_605497:
identityЂ dense_18/StatefulPartitionedCallЂ dense_19/StatefulPartitionedCallЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂlstm_15/StatefulPartitionedCallЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЁ
lstm_15/StatefulPartitionedCallStatefulPartitionedCallinputslstm_15_605454lstm_15_605456lstm_15_605458*
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6054532!
lstm_15/StatefulPartitionedCallЖ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_18_605473dense_18_605475*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_6054722"
 dense_18/StatefulPartitionedCallЗ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_605495dense_19_605497*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_6054942"
 dense_19/StatefulPartitionedCallў
reshape_9/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_6055132
reshape_9/PartitionedCallЮ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_15_605454*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЎ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_605497*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mul
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall0^dense_19/bias/Regularizer/Square/ReadVariableOp ^lstm_15/StatefulPartitionedCall>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
У
while_cond_604854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_604854___redundant_placeholder04
0while_while_cond_604854___redundant_placeholder14
0while_while_cond_604854___redundant_placeholder24
0while_while_cond_604854___redundant_placeholder3
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
а
Љ
C__inference_lstm_15_layer_call_and_return_conditional_losses_607298
inputs_0=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileF
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like}
lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout/ConstГ
lstm_cell_15/dropout/MulMullstm_cell_15/ones_like:output:0#lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul
lstm_cell_15/dropout/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout/Shapeј
1lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2№рю23
1lstm_cell_15/dropout/random_uniform/RandomUniform
#lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_15/dropout/GreaterEqual/yђ
!lstm_cell_15/dropout/GreaterEqualGreaterEqual:lstm_cell_15/dropout/random_uniform/RandomUniform:output:0,lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_15/dropout/GreaterEqualІ
lstm_cell_15/dropout/CastCast%lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/CastЎ
lstm_cell_15/dropout/Mul_1Mullstm_cell_15/dropout/Mul:z:0lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul_1
lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_1/ConstЙ
lstm_cell_15/dropout_1/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul
lstm_cell_15/dropout_1/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_1/Shapeў
3lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2хР25
3lstm_cell_15/dropout_1/random_uniform/RandomUniform
%lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_1/GreaterEqual/yњ
#lstm_cell_15/dropout_1/GreaterEqualGreaterEqual<lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_1/GreaterEqualЌ
lstm_cell_15/dropout_1/CastCast'lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/CastЖ
lstm_cell_15/dropout_1/Mul_1Mullstm_cell_15/dropout_1/Mul:z:0lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul_1
lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_2/ConstЙ
lstm_cell_15/dropout_2/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul
lstm_cell_15/dropout_2/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_2/Shapeў
3lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2рх25
3lstm_cell_15/dropout_2/random_uniform/RandomUniform
%lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_2/GreaterEqual/yњ
#lstm_cell_15/dropout_2/GreaterEqualGreaterEqual<lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_2/GreaterEqualЌ
lstm_cell_15/dropout_2/CastCast'lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/CastЖ
lstm_cell_15/dropout_2/Mul_1Mullstm_cell_15/dropout_2/Mul:z:0lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul_1
lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_3/ConstЙ
lstm_cell_15/dropout_3/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul
lstm_cell_15/dropout_3/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_3/Shapeў
3lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ОГе25
3lstm_cell_15/dropout_3/random_uniform/RandomUniform
%lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_3/GreaterEqual/yњ
#lstm_cell_15/dropout_3/GreaterEqualGreaterEqual<lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_3/GreaterEqualЌ
lstm_cell_15/dropout_3/CastCast'lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/CastЖ
lstm_cell_15/dropout_3/Mul_1Mullstm_cell_15/dropout_3/Mul:z:0lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul_1~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0 lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0 lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0 lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_607133*
condR
while_cond_607132*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ћВ
Є	
while_body_605726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
 while/lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_15/dropout/ConstЫ
while/lstm_cell_15/dropout/MulMul%while/lstm_cell_15/ones_like:output:0)while/lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_15/dropout/Mul
 while/lstm_cell_15/dropout/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_15/dropout/Shape
7while/lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2гБ29
7while/lstm_cell_15/dropout/random_uniform/RandomUniform
)while/lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_15/dropout/GreaterEqual/y
'while/lstm_cell_15/dropout/GreaterEqualGreaterEqual@while/lstm_cell_15/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_15/dropout/GreaterEqualИ
while/lstm_cell_15/dropout/CastCast+while/lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_15/dropout/CastЦ
 while/lstm_cell_15/dropout/Mul_1Mul"while/lstm_cell_15/dropout/Mul:z:0#while/lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout/Mul_1
"while/lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_1/Constб
 while/lstm_cell_15/dropout_1/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_1/Mul
"while/lstm_cell_15/dropout_1/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_1/Shape
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2њкл2;
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_1/GreaterEqual/y
)while/lstm_cell_15/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_1/GreaterEqualО
!while/lstm_cell_15/dropout_1/CastCast-while/lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_1/CastЮ
"while/lstm_cell_15/dropout_1/Mul_1Mul$while/lstm_cell_15/dropout_1/Mul:z:0%while/lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_1/Mul_1
"while/lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_2/Constб
 while/lstm_cell_15/dropout_2/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_2/Mul
"while/lstm_cell_15/dropout_2/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_2/Shape
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2юы2;
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_2/GreaterEqual/y
)while/lstm_cell_15/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_2/GreaterEqualО
!while/lstm_cell_15/dropout_2/CastCast-while/lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_2/CastЮ
"while/lstm_cell_15/dropout_2/Mul_1Mul$while/lstm_cell_15/dropout_2/Mul:z:0%while/lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_2/Mul_1
"while/lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_3/Constб
 while/lstm_cell_15/dropout_3/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_3/Mul
"while/lstm_cell_15/dropout_3/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_3/Shape
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2яњ2;
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_3/GreaterEqual/y
)while/lstm_cell_15/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_3/GreaterEqualО
!while/lstm_cell_15/dropout_3/CastCast-while/lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_3/CastЮ
"while/lstm_cell_15/dropout_3/Mul_1Mul$while/lstm_cell_15/dropout_3/Mul:z:0%while/lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_3/Mul_1
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Є
while/lstm_cell_15/mulMulwhile_placeholder_2$while/lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЊ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2&while/lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Њ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2&while/lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Њ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2&while/lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
ћВ
Є	
while_body_607683
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
 while/lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_15/dropout/ConstЫ
while/lstm_cell_15/dropout/MulMul%while/lstm_cell_15/ones_like:output:0)while/lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_15/dropout/Mul
 while/lstm_cell_15/dropout/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_15/dropout/Shape
7while/lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2зЌ29
7while/lstm_cell_15/dropout/random_uniform/RandomUniform
)while/lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_15/dropout/GreaterEqual/y
'while/lstm_cell_15/dropout/GreaterEqualGreaterEqual@while/lstm_cell_15/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_15/dropout/GreaterEqualИ
while/lstm_cell_15/dropout/CastCast+while/lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_15/dropout/CastЦ
 while/lstm_cell_15/dropout/Mul_1Mul"while/lstm_cell_15/dropout/Mul:z:0#while/lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout/Mul_1
"while/lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_1/Constб
 while/lstm_cell_15/dropout_1/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_1/Mul
"while/lstm_cell_15/dropout_1/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_1/Shape
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Е2;
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_1/GreaterEqual/y
)while/lstm_cell_15/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_1/GreaterEqualО
!while/lstm_cell_15/dropout_1/CastCast-while/lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_1/CastЮ
"while/lstm_cell_15/dropout_1/Mul_1Mul$while/lstm_cell_15/dropout_1/Mul:z:0%while/lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_1/Mul_1
"while/lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_2/Constб
 while/lstm_cell_15/dropout_2/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_2/Mul
"while/lstm_cell_15/dropout_2/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_2/Shape
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЯББ2;
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_2/GreaterEqual/y
)while/lstm_cell_15/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_2/GreaterEqualО
!while/lstm_cell_15/dropout_2/CastCast-while/lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_2/CastЮ
"while/lstm_cell_15/dropout_2/Mul_1Mul$while/lstm_cell_15/dropout_2/Mul:z:0%while/lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_2/Mul_1
"while/lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_3/Constб
 while/lstm_cell_15/dropout_3/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_3/Mul
"while/lstm_cell_15/dropout_3/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_3/Shape
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Їиљ2;
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_3/GreaterEqual/y
)while/lstm_cell_15/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_3/GreaterEqualО
!while/lstm_cell_15/dropout_3/CastCast-while/lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_3/CastЮ
"while/lstm_cell_15/dropout_3/Mul_1Mul$while/lstm_cell_15/dropout_3/Mul:z:0%while/lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_3/Mul_1
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Є
while/lstm_cell_15/mulMulwhile_placeholder_2$while/lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЊ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2&while/lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Њ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2&while/lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Њ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2&while/lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
и+
Њ
H__inference_sequential_6_layer_call_and_return_conditional_losses_605955

inputs!
lstm_15_605924:	
lstm_15_605926:	!
lstm_15_605928:	 !
dense_18_605931:  
dense_18_605933: !
dense_19_605936: 
dense_19_605938:
identityЂ dense_18/StatefulPartitionedCallЂ dense_19/StatefulPartitionedCallЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂlstm_15/StatefulPartitionedCallЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЁ
lstm_15/StatefulPartitionedCallStatefulPartitionedCallinputslstm_15_605924lstm_15_605926lstm_15_605928*
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6058912!
lstm_15/StatefulPartitionedCallЖ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_18_605931dense_18_605933*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_6054722"
 dense_18/StatefulPartitionedCallЗ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_605936dense_19_605938*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_6054942"
 dense_19/StatefulPartitionedCallў
reshape_9/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_6055132
reshape_9/PartitionedCallЮ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_15_605924*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЎ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_605938*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mul
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall0^dense_19/bias/Regularizer/Square/ReadVariableOp ^lstm_15/StatefulPartitionedCall>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
Е
(__inference_lstm_15_layer_call_fn_607892

inputs
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6058912
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
Ж
і
-__inference_lstm_cell_15_layer_call_fn_608206

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallУ
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6047772
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
м
Ъ
__inference_loss_fn_1_608217Y
Flstm_15_lstm_cell_15_kernel_regularizer_square_readvariableop_resource:	
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_15_lstm_cell_15_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muly
IdentityIdentity/lstm_15/lstm_cell_15/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp
Ц

у
lstm_15_while_cond_606510,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1D
@lstm_15_while_lstm_15_while_cond_606510___redundant_placeholder0D
@lstm_15_while_lstm_15_while_cond_606510___redundant_placeholder1D
@lstm_15_while_lstm_15_while_cond_606510___redundant_placeholder2D
@lstm_15_while_lstm_15_while_cond_606510___redundant_placeholder3
lstm_15_while_identity

lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2
lstm_15/while/Lessu
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_15/while/Identity"9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
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
у	
Ї
-__inference_sequential_6_layer_call_fn_605545
input_7
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
H__inference_sequential_6_layer_call_and_return_conditional_losses_6055282
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
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
Ї
Є	
while_body_606858
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Ѕ
while/lstm_cell_15/mulMulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЉ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Љ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Љ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
ь
Ї
D__inference_dense_19_layer_call_and_return_conditional_losses_607934

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_19/bias/Regularizer/Square/ReadVariableOp
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
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_19/bias/Regularizer/Square/ReadVariableOp*"
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
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
р	
І
-__inference_sequential_6_layer_call_fn_606742

inputs
unknown:	
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
H__inference_sequential_6_layer_call_and_return_conditional_losses_6059552
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

ѕ
D__inference_dense_18_layer_call_and_return_conditional_losses_605472

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
О
З
(__inference_lstm_15_layer_call_fn_607870
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	 
identityЂStatefulPartitionedCall
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6049302
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
оR
ы
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608059

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2,
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
е
У
while_cond_607682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_607682___redundant_placeholder04
0while_while_cond_607682___redundant_placeholder14
0while_while_cond_607682___redundant_placeholder24
0while_while_cond_607682___redundant_placeholder3
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
Књ
ь
!__inference__wrapped_model_604420
input_7R
?sequential_6_lstm_15_lstm_cell_15_split_readvariableop_resource:	P
Asequential_6_lstm_15_lstm_cell_15_split_1_readvariableop_resource:	L
9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource:	 F
4sequential_6_dense_18_matmul_readvariableop_resource:  C
5sequential_6_dense_18_biasadd_readvariableop_resource: F
4sequential_6_dense_19_matmul_readvariableop_resource: C
5sequential_6_dense_19_biasadd_readvariableop_resource:
identityЂ,sequential_6/dense_18/BiasAdd/ReadVariableOpЂ+sequential_6/dense_18/MatMul/ReadVariableOpЂ,sequential_6/dense_19/BiasAdd/ReadVariableOpЂ+sequential_6/dense_19/MatMul/ReadVariableOpЂ0sequential_6/lstm_15/lstm_cell_15/ReadVariableOpЂ2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_1Ђ2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_2Ђ2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_3Ђ6sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOpЂ8sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOpЂsequential_6/lstm_15/whileo
sequential_6/lstm_15/ShapeShapeinput_7*
T0*
_output_shapes
:2
sequential_6/lstm_15/Shape
(sequential_6/lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_15/strided_slice/stackЂ
*sequential_6/lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_15/strided_slice/stack_1Ђ
*sequential_6/lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_15/strided_slice/stack_2р
"sequential_6/lstm_15/strided_sliceStridedSlice#sequential_6/lstm_15/Shape:output:01sequential_6/lstm_15/strided_slice/stack:output:03sequential_6/lstm_15/strided_slice/stack_1:output:03sequential_6/lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_15/strided_slice
 sequential_6/lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_6/lstm_15/zeros/mul/yР
sequential_6/lstm_15/zeros/mulMul+sequential_6/lstm_15/strided_slice:output:0)sequential_6/lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_15/zeros/mul
!sequential_6/lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_6/lstm_15/zeros/Less/yЛ
sequential_6/lstm_15/zeros/LessLess"sequential_6/lstm_15/zeros/mul:z:0*sequential_6/lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_15/zeros/Less
#sequential_6/lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_6/lstm_15/zeros/packed/1з
!sequential_6/lstm_15/zeros/packedPack+sequential_6/lstm_15/strided_slice:output:0,sequential_6/lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_15/zeros/packed
 sequential_6/lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_15/zeros/ConstЩ
sequential_6/lstm_15/zerosFill*sequential_6/lstm_15/zeros/packed:output:0)sequential_6/lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_6/lstm_15/zeros
"sequential_6/lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_6/lstm_15/zeros_1/mul/yЦ
 sequential_6/lstm_15/zeros_1/mulMul+sequential_6/lstm_15/strided_slice:output:0+sequential_6/lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_15/zeros_1/mul
#sequential_6/lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_6/lstm_15/zeros_1/Less/yУ
!sequential_6/lstm_15/zeros_1/LessLess$sequential_6/lstm_15/zeros_1/mul:z:0,sequential_6/lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_15/zeros_1/Less
%sequential_6/lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_6/lstm_15/zeros_1/packed/1н
#sequential_6/lstm_15/zeros_1/packedPack+sequential_6/lstm_15/strided_slice:output:0.sequential_6/lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_15/zeros_1/packed
"sequential_6/lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_15/zeros_1/Constб
sequential_6/lstm_15/zeros_1Fill,sequential_6/lstm_15/zeros_1/packed:output:0+sequential_6/lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_6/lstm_15/zeros_1
#sequential_6/lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_15/transpose/permК
sequential_6/lstm_15/transpose	Transposeinput_7,sequential_6/lstm_15/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_6/lstm_15/transpose
sequential_6/lstm_15/Shape_1Shape"sequential_6/lstm_15/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_15/Shape_1Ђ
*sequential_6/lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_15/strided_slice_1/stackІ
,sequential_6/lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_15/strided_slice_1/stack_1І
,sequential_6/lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_15/strided_slice_1/stack_2ь
$sequential_6/lstm_15/strided_slice_1StridedSlice%sequential_6/lstm_15/Shape_1:output:03sequential_6/lstm_15/strided_slice_1/stack:output:05sequential_6/lstm_15/strided_slice_1/stack_1:output:05sequential_6/lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_15/strided_slice_1Џ
0sequential_6/lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ22
0sequential_6/lstm_15/TensorArrayV2/element_shape
"sequential_6/lstm_15/TensorArrayV2TensorListReserve9sequential_6/lstm_15/TensorArrayV2/element_shape:output:0-sequential_6/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_15/TensorArrayV2щ
Jsequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2L
Jsequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeЬ
<sequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_15/transpose:y:0Ssequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensorЂ
*sequential_6/lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_15/strided_slice_2/stackІ
,sequential_6/lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_15/strided_slice_2/stack_1І
,sequential_6/lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_15/strided_slice_2/stack_2њ
$sequential_6/lstm_15/strided_slice_2StridedSlice"sequential_6/lstm_15/transpose:y:03sequential_6/lstm_15/strided_slice_2/stack:output:05sequential_6/lstm_15/strided_slice_2/stack_1:output:05sequential_6/lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2&
$sequential_6/lstm_15/strided_slice_2Й
1sequential_6/lstm_15/lstm_cell_15/ones_like/ShapeShape#sequential_6/lstm_15/zeros:output:0*
T0*
_output_shapes
:23
1sequential_6/lstm_15/lstm_cell_15/ones_like/ShapeЋ
1sequential_6/lstm_15/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1sequential_6/lstm_15/lstm_cell_15/ones_like/Const
+sequential_6/lstm_15/lstm_cell_15/ones_likeFill:sequential_6/lstm_15/lstm_cell_15/ones_like/Shape:output:0:sequential_6/lstm_15/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/ones_likeЈ
1sequential_6/lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_15/lstm_cell_15/split/split_dimё
6sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOpReadVariableOp?sequential_6_lstm_15_lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOpЏ
'sequential_6/lstm_15/lstm_cell_15/splitSplit:sequential_6/lstm_15/lstm_cell_15/split/split_dim:output:0>sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2)
'sequential_6/lstm_15/lstm_cell_15/splitё
(sequential_6/lstm_15/lstm_cell_15/MatMulMatMul-sequential_6/lstm_15/strided_slice_2:output:00sequential_6/lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_6/lstm_15/lstm_cell_15/MatMulѕ
*sequential_6/lstm_15/lstm_cell_15/MatMul_1MatMul-sequential_6/lstm_15/strided_slice_2:output:00sequential_6/lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_1ѕ
*sequential_6/lstm_15/lstm_cell_15/MatMul_2MatMul-sequential_6/lstm_15/strided_slice_2:output:00sequential_6/lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_2ѕ
*sequential_6/lstm_15/lstm_cell_15/MatMul_3MatMul-sequential_6/lstm_15/strided_slice_2:output:00sequential_6/lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_3Ќ
3sequential_6/lstm_15/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_6/lstm_15/lstm_cell_15/split_1/split_dimѓ
8sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOpReadVariableOpAsequential_6_lstm_15_lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02:
8sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOpЇ
)sequential_6/lstm_15/lstm_cell_15/split_1Split<sequential_6/lstm_15/lstm_cell_15/split_1/split_dim:output:0@sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2+
)sequential_6/lstm_15/lstm_cell_15/split_1ћ
)sequential_6/lstm_15/lstm_cell_15/BiasAddBiasAdd2sequential_6/lstm_15/lstm_cell_15/MatMul:product:02sequential_6/lstm_15/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_6/lstm_15/lstm_cell_15/BiasAdd
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_1BiasAdd4sequential_6/lstm_15/lstm_cell_15/MatMul_1:product:02sequential_6/lstm_15/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_1
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_2BiasAdd4sequential_6/lstm_15/lstm_cell_15/MatMul_2:product:02sequential_6/lstm_15/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_2
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_3BiasAdd4sequential_6/lstm_15/lstm_cell_15/MatMul_3:product:02sequential_6/lstm_15/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/BiasAdd_3т
%sequential_6/lstm_15/lstm_cell_15/mulMul#sequential_6/lstm_15/zeros:output:04sequential_6/lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_6/lstm_15/lstm_cell_15/mulц
'sequential_6/lstm_15/lstm_cell_15/mul_1Mul#sequential_6/lstm_15/zeros:output:04sequential_6/lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_1ц
'sequential_6/lstm_15/lstm_cell_15/mul_2Mul#sequential_6/lstm_15/zeros:output:04sequential_6/lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_2ц
'sequential_6/lstm_15/lstm_cell_15/mul_3Mul#sequential_6/lstm_15/zeros:output:04sequential_6/lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_3п
0sequential_6/lstm_15/lstm_cell_15/ReadVariableOpReadVariableOp9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype022
0sequential_6/lstm_15/lstm_cell_15/ReadVariableOpП
5sequential_6/lstm_15/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_6/lstm_15/lstm_cell_15/strided_slice/stackУ
7sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_1У
7sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_2Ш
/sequential_6/lstm_15/lstm_cell_15/strided_sliceStridedSlice8sequential_6/lstm_15/lstm_cell_15/ReadVariableOp:value:0>sequential_6/lstm_15/lstm_cell_15/strided_slice/stack:output:0@sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_1:output:0@sequential_6/lstm_15/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_6/lstm_15/lstm_cell_15/strided_sliceљ
*sequential_6/lstm_15/lstm_cell_15/MatMul_4MatMul)sequential_6/lstm_15/lstm_cell_15/mul:z:08sequential_6/lstm_15/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_4ѓ
%sequential_6/lstm_15/lstm_cell_15/addAddV22sequential_6/lstm_15/lstm_cell_15/BiasAdd:output:04sequential_6/lstm_15/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_6/lstm_15/lstm_cell_15/addО
)sequential_6/lstm_15/lstm_cell_15/SigmoidSigmoid)sequential_6/lstm_15/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_6/lstm_15/lstm_cell_15/Sigmoidу
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_1ReadVariableOp9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype024
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_1У
7sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stackЧ
9sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_1Ч
9sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_2д
1sequential_6/lstm_15/lstm_cell_15/strided_slice_1StridedSlice:sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_1:value:0@sequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_1:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_6/lstm_15/lstm_cell_15/strided_slice_1§
*sequential_6/lstm_15/lstm_cell_15/MatMul_5MatMul+sequential_6/lstm_15/lstm_cell_15/mul_1:z:0:sequential_6/lstm_15/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_5љ
'sequential_6/lstm_15/lstm_cell_15/add_1AddV24sequential_6/lstm_15/lstm_cell_15/BiasAdd_1:output:04sequential_6/lstm_15/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/add_1Ф
+sequential_6/lstm_15/lstm_cell_15/Sigmoid_1Sigmoid+sequential_6/lstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/Sigmoid_1у
'sequential_6/lstm_15/lstm_cell_15/mul_4Mul/sequential_6/lstm_15/lstm_cell_15/Sigmoid_1:y:0%sequential_6/lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_4у
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_2ReadVariableOp9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype024
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_2У
7sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stackЧ
9sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_1Ч
9sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_2д
1sequential_6/lstm_15/lstm_cell_15/strided_slice_2StridedSlice:sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_2:value:0@sequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_1:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_6/lstm_15/lstm_cell_15/strided_slice_2§
*sequential_6/lstm_15/lstm_cell_15/MatMul_6MatMul+sequential_6/lstm_15/lstm_cell_15/mul_2:z:0:sequential_6/lstm_15/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_6љ
'sequential_6/lstm_15/lstm_cell_15/add_2AddV24sequential_6/lstm_15/lstm_cell_15/BiasAdd_2:output:04sequential_6/lstm_15/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/add_2З
&sequential_6/lstm_15/lstm_cell_15/ReluRelu+sequential_6/lstm_15/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_6/lstm_15/lstm_cell_15/Relu№
'sequential_6/lstm_15/lstm_cell_15/mul_5Mul-sequential_6/lstm_15/lstm_cell_15/Sigmoid:y:04sequential_6/lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_5ч
'sequential_6/lstm_15/lstm_cell_15/add_3AddV2+sequential_6/lstm_15/lstm_cell_15/mul_4:z:0+sequential_6/lstm_15/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/add_3у
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_3ReadVariableOp9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype024
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_3У
7sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   29
7sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stackЧ
9sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_1Ч
9sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_2д
1sequential_6/lstm_15/lstm_cell_15/strided_slice_3StridedSlice:sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_3:value:0@sequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_1:output:0Bsequential_6/lstm_15/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1sequential_6/lstm_15/lstm_cell_15/strided_slice_3§
*sequential_6/lstm_15/lstm_cell_15/MatMul_7MatMul+sequential_6/lstm_15/lstm_cell_15/mul_3:z:0:sequential_6/lstm_15/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_6/lstm_15/lstm_cell_15/MatMul_7љ
'sequential_6/lstm_15/lstm_cell_15/add_4AddV24sequential_6/lstm_15/lstm_cell_15/BiasAdd_3:output:04sequential_6/lstm_15/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/add_4Ф
+sequential_6/lstm_15/lstm_cell_15/Sigmoid_2Sigmoid+sequential_6/lstm_15/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_6/lstm_15/lstm_cell_15/Sigmoid_2Л
(sequential_6/lstm_15/lstm_cell_15/Relu_1Relu+sequential_6/lstm_15/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_6/lstm_15/lstm_cell_15/Relu_1є
'sequential_6/lstm_15/lstm_cell_15/mul_6Mul/sequential_6/lstm_15/lstm_cell_15/Sigmoid_2:y:06sequential_6/lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_6/lstm_15/lstm_cell_15/mul_6Й
2sequential_6/lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    24
2sequential_6/lstm_15/TensorArrayV2_1/element_shape
$sequential_6/lstm_15/TensorArrayV2_1TensorListReserve;sequential_6/lstm_15/TensorArrayV2_1/element_shape:output:0-sequential_6/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_6/lstm_15/TensorArrayV2_1x
sequential_6/lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_6/lstm_15/timeЉ
-sequential_6/lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-sequential_6/lstm_15/while/maximum_iterations
'sequential_6/lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_15/while/loop_counterМ
sequential_6/lstm_15/whileWhile0sequential_6/lstm_15/while/loop_counter:output:06sequential_6/lstm_15/while/maximum_iterations:output:0"sequential_6/lstm_15/time:output:0-sequential_6/lstm_15/TensorArrayV2_1:handle:0#sequential_6/lstm_15/zeros:output:0%sequential_6/lstm_15/zeros_1:output:0-sequential_6/lstm_15/strided_slice_1:output:0Lsequential_6/lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_6_lstm_15_lstm_cell_15_split_readvariableop_resourceAsequential_6_lstm_15_lstm_cell_15_split_1_readvariableop_resource9sequential_6_lstm_15_lstm_cell_15_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_6_lstm_15_while_body_604271*2
cond*R(
&sequential_6_lstm_15_while_cond_604270*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_6/lstm_15/whileп
Esequential_6/lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2G
Esequential_6/lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeМ
7sequential_6/lstm_15/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_15/while:output:3Nsequential_6/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype029
7sequential_6/lstm_15/TensorArrayV2Stack/TensorListStackЋ
*sequential_6/lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2,
*sequential_6/lstm_15/strided_slice_3/stackІ
,sequential_6/lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_15/strided_slice_3/stack_1І
,sequential_6/lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_15/strided_slice_3/stack_2
$sequential_6/lstm_15/strided_slice_3StridedSlice@sequential_6/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_15/strided_slice_3/stack:output:05sequential_6/lstm_15/strided_slice_3/stack_1:output:05sequential_6/lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2&
$sequential_6/lstm_15/strided_slice_3Ѓ
%sequential_6/lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_15/transpose_1/permљ
 sequential_6/lstm_15/transpose_1	Transpose@sequential_6/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2"
 sequential_6/lstm_15/transpose_1
sequential_6/lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_15/runtimeЯ
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOpм
sequential_6/dense_18/MatMulMatMul-sequential_6/lstm_15/strided_slice_3:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_6/dense_18/MatMulЮ
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOpй
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_6/dense_18/BiasAdd
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_6/dense_18/ReluЯ
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOpз
sequential_6/dense_19/MatMulMatMul(sequential_6/dense_18/Relu:activations:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_6/dense_19/MatMulЮ
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOpй
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_6/dense_19/BiasAdd
sequential_6/reshape_9/ShapeShape&sequential_6/dense_19/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_6/reshape_9/ShapeЂ
*sequential_6/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/reshape_9/strided_slice/stackІ
,sequential_6/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/reshape_9/strided_slice/stack_1І
,sequential_6/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/reshape_9/strided_slice/stack_2ь
$sequential_6/reshape_9/strided_sliceStridedSlice%sequential_6/reshape_9/Shape:output:03sequential_6/reshape_9/strided_slice/stack:output:05sequential_6/reshape_9/strided_slice/stack_1:output:05sequential_6/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/reshape_9/strided_slice
&sequential_6/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_6/reshape_9/Reshape/shape/1
&sequential_6/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_6/reshape_9/Reshape/shape/2
$sequential_6/reshape_9/Reshape/shapePack-sequential_6/reshape_9/strided_slice:output:0/sequential_6/reshape_9/Reshape/shape/1:output:0/sequential_6/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_6/reshape_9/Reshape/shapeи
sequential_6/reshape_9/ReshapeReshape&sequential_6/dense_19/BiasAdd:output:0-sequential_6/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_6/reshape_9/Reshape
IdentityIdentity'sequential_6/reshape_9/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identityы
NoOpNoOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp1^sequential_6/lstm_15/lstm_cell_15/ReadVariableOp3^sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_13^sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_23^sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_37^sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOp9^sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOp^sequential_6/lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp2d
0sequential_6/lstm_15/lstm_cell_15/ReadVariableOp0sequential_6/lstm_15/lstm_cell_15/ReadVariableOp2h
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_12sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_12h
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_22sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_22h
2sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_32sequential_6/lstm_15/lstm_cell_15/ReadVariableOp_32p
6sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOp6sequential_6/lstm_15/lstm_cell_15/split/ReadVariableOp2t
8sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOp8sequential_6/lstm_15/lstm_cell_15/split_1/ReadVariableOp28
sequential_6/lstm_15/whilesequential_6/lstm_15/while:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
Ж
і
-__inference_lstm_cell_15_layer_call_fn_608189

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:	 
identity

identity_1

identity_2ЂStatefulPartitionedCallУ
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6045442
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
л+
Ћ
H__inference_sequential_6_layer_call_and_return_conditional_losses_606025
input_7!
lstm_15_605994:	
lstm_15_605996:	!
lstm_15_605998:	 !
dense_18_606001:  
dense_18_606003: !
dense_19_606006: 
dense_19_606008:
identityЂ dense_18/StatefulPartitionedCallЂ dense_19/StatefulPartitionedCallЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂlstm_15/StatefulPartitionedCallЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ
lstm_15/StatefulPartitionedCallStatefulPartitionedCallinput_7lstm_15_605994lstm_15_605996lstm_15_605998*
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
GPU 2J 8 *L
fGRE
C__inference_lstm_15_layer_call_and_return_conditional_losses_6054532!
lstm_15/StatefulPartitionedCallЖ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_18_606001dense_18_606003*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_6054722"
 dense_18/StatefulPartitionedCallЗ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_606006dense_19_606008*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_6054942"
 dense_19/StatefulPartitionedCallў
reshape_9/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_6055132
reshape_9/PartitionedCallЮ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_15_605994*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЎ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_606008*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mul
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЈ
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall0^dense_19/bias/Regularizer/Square/ReadVariableOp ^lstm_15/StatefulPartitionedCall>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
Г	

$__inference_signature_wrapper_606098
input_7
unknown:	
	unknown_0:	
	unknown_1:	 
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
!__inference__wrapped_model_6044202
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
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7
и%
у
while_body_604855
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_15_604879_0:	*
while_lstm_cell_15_604881_0:	.
while_lstm_cell_15_604883_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_15_604879:	(
while_lstm_cell_15_604881:	,
while_lstm_cell_15_604883:	 Ђ*while/lstm_cell_15/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_604879_0while_lstm_cell_15_604881_0while_lstm_cell_15_604883_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_6047772,
*while/lstm_cell_15/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Є
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_15/StatefulPartitionedCall*"
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
while_lstm_cell_15_604879while_lstm_cell_15_604879_0"8
while_lstm_cell_15_604881while_lstm_cell_15_604881_0"8
while_lstm_cell_15_604883while_lstm_cell_15_604883_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
Ц

у
lstm_15_while_cond_606207,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1D
@lstm_15_while_lstm_15_while_cond_606207___redundant_placeholder0D
@lstm_15_while_lstm_15_while_cond_606207___redundant_placeholder1D
@lstm_15_while_lstm_15_while_cond_606207___redundant_placeholder2D
@lstm_15_while_lstm_15_while_cond_606207___redundant_placeholder3
lstm_15_while_identity

lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2
lstm_15/while/Lessu
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_15/while/Identity"9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
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
while_cond_607132
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_607132___redundant_placeholder04
0while_while_cond_607132___redundant_placeholder14
0while_while_cond_607132___redundant_placeholder24
0while_while_cond_607132___redundant_placeholder3
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
Ц
F
*__inference_reshape_9_layer_call_fn_607961

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
E__inference_reshape_9_layer_call_and_return_conditional_losses_6055132
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
Ї
Є	
while_body_607408
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Ѕ
while/lstm_cell_15/mulMulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЉ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Љ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Љ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2%while/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_605513

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

ѕ
D__inference_dense_18_layer_call_and_return_conditional_losses_607903

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
Ё|

"__inference__traced_restore_608418
file_prefix2
 assignvariableop_dense_18_kernel:  .
 assignvariableop_1_dense_18_bias: 4
"assignvariableop_2_dense_19_kernel: .
 assignvariableop_3_dense_19_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_15_lstm_cell_15_kernel:	L
9assignvariableop_10_lstm_15_lstm_cell_15_recurrent_kernel:	 <
-assignvariableop_11_lstm_15_lstm_cell_15_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_18_kernel_m:  6
(assignvariableop_15_adam_dense_18_bias_m: <
*assignvariableop_16_adam_dense_19_kernel_m: 6
(assignvariableop_17_adam_dense_19_bias_m:I
6assignvariableop_18_adam_lstm_15_lstm_cell_15_kernel_m:	S
@assignvariableop_19_adam_lstm_15_lstm_cell_15_recurrent_kernel_m:	 C
4assignvariableop_20_adam_lstm_15_lstm_cell_15_bias_m:	<
*assignvariableop_21_adam_dense_18_kernel_v:  6
(assignvariableop_22_adam_dense_18_bias_v: <
*assignvariableop_23_adam_dense_19_kernel_v: 6
(assignvariableop_24_adam_dense_19_bias_v:I
6assignvariableop_25_adam_lstm_15_lstm_cell_15_kernel_v:	S
@assignvariableop_26_adam_lstm_15_lstm_cell_15_recurrent_kernel_v:	 C
4assignvariableop_27_adam_lstm_15_lstm_cell_15_bias_v:	
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
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_15_lstm_cell_15_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10С
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_15_lstm_cell_15_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_15_lstm_cell_15_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_18_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15А
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_18_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_19_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_19_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18О
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_15_lstm_cell_15_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ш
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_15_lstm_cell_15_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20М
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_15_lstm_cell_15_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_18_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_18_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_19_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_19_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25О
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_15_lstm_cell_15_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ш
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_15_lstm_cell_15_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27М
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_15_lstm_cell_15_bias_vIdentity_27:output:0"/device:CPU:0*
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
е
У
while_cond_606857
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_606857___redundant_placeholder04
0while_while_cond_606857___redundant_placeholder14
0while_while_cond_606857___redundant_placeholder24
0while_while_cond_606857___redundant_placeholder3
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
љВ
Є	
while_body_607133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_15_split_readvariableop_resource_0:	C
4while_lstm_cell_15_split_1_readvariableop_resource_0:	?
,while_lstm_cell_15_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_15_split_readvariableop_resource:	A
2while_lstm_cell_15_split_1_readvariableop_resource:	=
*while_lstm_cell_15_readvariableop_resource:	 Ђ!while/lstm_cell_15/ReadVariableOpЂ#while/lstm_cell_15/ReadVariableOp_1Ђ#while/lstm_cell_15/ReadVariableOp_2Ђ#while/lstm_cell_15/ReadVariableOp_3Ђ'while/lstm_cell_15/split/ReadVariableOpЂ)while/lstm_cell_15/split_1/ReadVariableOpУ
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
"while/lstm_cell_15/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/lstm_cell_15/ones_like/Shape
"while/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_15/ones_like/Constа
while/lstm_cell_15/ones_likeFill+while/lstm_cell_15/ones_like/Shape:output:0+while/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ones_like
 while/lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2"
 while/lstm_cell_15/dropout/ConstЫ
while/lstm_cell_15/dropout/MulMul%while/lstm_cell_15/ones_like:output:0)while/lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_15/dropout/Mul
 while/lstm_cell_15/dropout/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_15/dropout/Shape
7while/lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Њ29
7while/lstm_cell_15/dropout/random_uniform/RandomUniform
)while/lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2+
)while/lstm_cell_15/dropout/GreaterEqual/y
'while/lstm_cell_15/dropout/GreaterEqualGreaterEqual@while/lstm_cell_15/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'while/lstm_cell_15/dropout/GreaterEqualИ
while/lstm_cell_15/dropout/CastCast+while/lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2!
while/lstm_cell_15/dropout/CastЦ
 while/lstm_cell_15/dropout/Mul_1Mul"while/lstm_cell_15/dropout/Mul:z:0#while/lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout/Mul_1
"while/lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_1/Constб
 while/lstm_cell_15/dropout_1/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_1/Mul
"while/lstm_cell_15/dropout_1/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_1/Shape
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Пи_2;
9while/lstm_cell_15/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_1/GreaterEqual/y
)while/lstm_cell_15/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_1/GreaterEqualО
!while/lstm_cell_15/dropout_1/CastCast-while/lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_1/CastЮ
"while/lstm_cell_15/dropout_1/Mul_1Mul$while/lstm_cell_15/dropout_1/Mul:z:0%while/lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_1/Mul_1
"while/lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_2/Constб
 while/lstm_cell_15/dropout_2/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_2/Mul
"while/lstm_cell_15/dropout_2/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_2/Shape
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЌW2;
9while/lstm_cell_15/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_2/GreaterEqual/y
)while/lstm_cell_15/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_2/GreaterEqualО
!while/lstm_cell_15/dropout_2/CastCast-while/lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_2/CastЮ
"while/lstm_cell_15/dropout_2/Mul_1Mul$while/lstm_cell_15/dropout_2/Mul:z:0%while/lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_2/Mul_1
"while/lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2$
"while/lstm_cell_15/dropout_3/Constб
 while/lstm_cell_15/dropout_3/MulMul%while/lstm_cell_15/ones_like:output:0+while/lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_15/dropout_3/Mul
"while/lstm_cell_15/dropout_3/ShapeShape%while/lstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_15/dropout_3/Shape
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2Оьч2;
9while/lstm_cell_15/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2-
+while/lstm_cell_15/dropout_3/GreaterEqual/y
)while/lstm_cell_15/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_15/dropout_3/GreaterEqualО
!while/lstm_cell_15/dropout_3/CastCast-while/lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_15/dropout_3/CastЮ
"while/lstm_cell_15/dropout_3/Mul_1Mul$while/lstm_cell_15/dropout_3/Mul:z:0%while/lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_15/dropout_3/Mul_1
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimЦ
'while/lstm_cell_15/split/ReadVariableOpReadVariableOp2while_lstm_cell_15_split_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/lstm_cell_15/split/ReadVariableOpѓ
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0/while/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_15/splitЧ
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMulЫ
while/lstm_cell_15/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_1Ы
while/lstm_cell_15/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_2Ы
while/lstm_cell_15/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_3
$while/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_15/split_1/split_dimШ
)while/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_15_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_15/split_1/ReadVariableOpы
while/lstm_cell_15/split_1Split-while/lstm_cell_15/split_1/split_dim:output:01while/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_15/split_1П
while/lstm_cell_15/BiasAddBiasAdd#while/lstm_cell_15/MatMul:product:0#while/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAddХ
while/lstm_cell_15/BiasAdd_1BiasAdd%while/lstm_cell_15/MatMul_1:product:0#while/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_1Х
while/lstm_cell_15/BiasAdd_2BiasAdd%while/lstm_cell_15/MatMul_2:product:0#while/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_2Х
while/lstm_cell_15/BiasAdd_3BiasAdd%while/lstm_cell_15/MatMul_3:product:0#while/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/BiasAdd_3Є
while/lstm_cell_15/mulMulwhile_placeholder_2$while/lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mulЊ
while/lstm_cell_15/mul_1Mulwhile_placeholder_2&while/lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_1Њ
while/lstm_cell_15/mul_2Mulwhile_placeholder_2&while/lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_2Њ
while/lstm_cell_15/mul_3Mulwhile_placeholder_2&while/lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_3Д
!while/lstm_cell_15/ReadVariableOpReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_15/ReadVariableOpЁ
&while/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_15/strided_slice/stackЅ
(while/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice/stack_1Ѕ
(while/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_15/strided_slice/stack_2ю
 while/lstm_cell_15/strided_sliceStridedSlice)while/lstm_cell_15/ReadVariableOp:value:0/while/lstm_cell_15/strided_slice/stack:output:01while/lstm_cell_15/strided_slice/stack_1:output:01while/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_15/strided_sliceН
while/lstm_cell_15/MatMul_4MatMulwhile/lstm_cell_15/mul:z:0)while/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_4З
while/lstm_cell_15/addAddV2#while/lstm_cell_15/BiasAdd:output:0%while/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add
while/lstm_cell_15/SigmoidSigmoidwhile/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/SigmoidИ
#while/lstm_cell_15/ReadVariableOp_1ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_1Ѕ
(while/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_15/strided_slice_1/stackЉ
*while/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_15/strided_slice_1/stack_1Љ
*while/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_1/stack_2њ
"while/lstm_cell_15/strided_slice_1StridedSlice+while/lstm_cell_15/ReadVariableOp_1:value:01while/lstm_cell_15/strided_slice_1/stack:output:03while/lstm_cell_15/strided_slice_1/stack_1:output:03while/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_1С
while/lstm_cell_15/MatMul_5MatMulwhile/lstm_cell_15/mul_1:z:0+while/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_5Н
while/lstm_cell_15/add_1AddV2%while/lstm_cell_15/BiasAdd_1:output:0%while/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_1Sigmoidwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_1Є
while/lstm_cell_15/mul_4Mul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_4И
#while/lstm_cell_15/ReadVariableOp_2ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_2Ѕ
(while/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_15/strided_slice_2/stackЉ
*while/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_15/strided_slice_2/stack_1Љ
*while/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_2/stack_2њ
"while/lstm_cell_15/strided_slice_2StridedSlice+while/lstm_cell_15/ReadVariableOp_2:value:01while/lstm_cell_15/strided_slice_2/stack:output:03while/lstm_cell_15/strided_slice_2/stack_1:output:03while/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_2С
while/lstm_cell_15/MatMul_6MatMulwhile/lstm_cell_15/mul_2:z:0+while/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_6Н
while/lstm_cell_15/add_2AddV2%while/lstm_cell_15/BiasAdd_2:output:0%while/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_2
while/lstm_cell_15/ReluReluwhile/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/ReluД
while/lstm_cell_15/mul_5Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_5Ћ
while/lstm_cell_15/add_3AddV2while/lstm_cell_15/mul_4:z:0while/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_3И
#while/lstm_cell_15/ReadVariableOp_3ReadVariableOp,while_lstm_cell_15_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_15/ReadVariableOp_3Ѕ
(while/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_15/strided_slice_3/stackЉ
*while/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_15/strided_slice_3/stack_1Љ
*while/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_15/strided_slice_3/stack_2њ
"while/lstm_cell_15/strided_slice_3StridedSlice+while/lstm_cell_15/ReadVariableOp_3:value:01while/lstm_cell_15/strided_slice_3/stack:output:03while/lstm_cell_15/strided_slice_3/stack_1:output:03while/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_15/strided_slice_3С
while/lstm_cell_15/MatMul_7MatMulwhile/lstm_cell_15/mul_3:z:0+while/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/MatMul_7Н
while/lstm_cell_15/add_4AddV2%while/lstm_cell_15/BiasAdd_3:output:0%while/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/add_4
while/lstm_cell_15/Sigmoid_2Sigmoidwhile/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/Relu_1И
while/lstm_cell_15/mul_6Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_15/mul_6р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_6:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5Ц

while/NoOpNoOp"^while/lstm_cell_15/ReadVariableOp$^while/lstm_cell_15/ReadVariableOp_1$^while/lstm_cell_15/ReadVariableOp_2$^while/lstm_cell_15/ReadVariableOp_3(^while/lstm_cell_15/split/ReadVariableOp*^while/lstm_cell_15/split_1/ReadVariableOp*"
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
*while_lstm_cell_15_readvariableop_resource,while_lstm_cell_15_readvariableop_resource_0"j
2while_lstm_cell_15_split_1_readvariableop_resource4while_lstm_cell_15_split_1_readvariableop_resource_0"f
0while_lstm_cell_15_split_readvariableop_resource2while_lstm_cell_15_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : 2F
!while/lstm_cell_15/ReadVariableOp!while/lstm_cell_15/ReadVariableOp2J
#while/lstm_cell_15/ReadVariableOp_1#while/lstm_cell_15/ReadVariableOp_12J
#while/lstm_cell_15/ReadVariableOp_2#while/lstm_cell_15/ReadVariableOp_22J
#while/lstm_cell_15/ReadVariableOp_3#while/lstm_cell_15/ReadVariableOp_32R
'while/lstm_cell_15/split/ReadVariableOp'while/lstm_cell_15/split/ReadVariableOp2V
)while/lstm_cell_15/split_1/ReadVariableOp)while/lstm_cell_15/split_1/ReadVariableOp: 
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
ь
Ї
D__inference_dense_19_layer_call_and_return_conditional_losses_605494

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/dense_19/bias/Regularizer/Square/ReadVariableOp
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
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityБ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_19/bias/Regularizer/Square/ReadVariableOp*"
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
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
оЁ
Ї
C__inference_lstm_15_layer_call_and_return_conditional_losses_605453

inputs=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileD
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_605320*
condR
while_cond_605319*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
йЯ
Ї
C__inference_lstm_15_layer_call_and_return_conditional_losses_607848

inputs=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileD
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like}
lstm_cell_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout/ConstГ
lstm_cell_15/dropout/MulMullstm_cell_15/ones_like:output:0#lstm_cell_15/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul
lstm_cell_15/dropout/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout/Shapeј
1lstm_cell_15/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_15/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ОЛм23
1lstm_cell_15/dropout/random_uniform/RandomUniform
#lstm_cell_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#lstm_cell_15/dropout/GreaterEqual/yђ
!lstm_cell_15/dropout/GreaterEqualGreaterEqual:lstm_cell_15/dropout/random_uniform/RandomUniform:output:0,lstm_cell_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_cell_15/dropout/GreaterEqualІ
lstm_cell_15/dropout/CastCast%lstm_cell_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/CastЎ
lstm_cell_15/dropout/Mul_1Mullstm_cell_15/dropout/Mul:z:0lstm_cell_15/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout/Mul_1
lstm_cell_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_1/ConstЙ
lstm_cell_15/dropout_1/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul
lstm_cell_15/dropout_1/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_1/Shapeў
3lstm_cell_15/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2гф25
3lstm_cell_15/dropout_1/random_uniform/RandomUniform
%lstm_cell_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_1/GreaterEqual/yњ
#lstm_cell_15/dropout_1/GreaterEqualGreaterEqual<lstm_cell_15/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_1/GreaterEqualЌ
lstm_cell_15/dropout_1/CastCast'lstm_cell_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/CastЖ
lstm_cell_15/dropout_1/Mul_1Mullstm_cell_15/dropout_1/Mul:z:0lstm_cell_15/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_1/Mul_1
lstm_cell_15/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_2/ConstЙ
lstm_cell_15/dropout_2/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul
lstm_cell_15/dropout_2/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_2/Shapeў
3lstm_cell_15/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2љЪ25
3lstm_cell_15/dropout_2/random_uniform/RandomUniform
%lstm_cell_15/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_2/GreaterEqual/yњ
#lstm_cell_15/dropout_2/GreaterEqualGreaterEqual<lstm_cell_15/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_2/GreaterEqualЌ
lstm_cell_15/dropout_2/CastCast'lstm_cell_15/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/CastЖ
lstm_cell_15/dropout_2/Mul_1Mullstm_cell_15/dropout_2/Mul:z:0lstm_cell_15/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_2/Mul_1
lstm_cell_15/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
lstm_cell_15/dropout_3/ConstЙ
lstm_cell_15/dropout_3/MulMullstm_cell_15/ones_like:output:0%lstm_cell_15/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul
lstm_cell_15/dropout_3/ShapeShapelstm_cell_15/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_15/dropout_3/Shapeў
3lstm_cell_15/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_15/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedв	*
seed2ЗЅ25
3lstm_cell_15/dropout_3/random_uniform/RandomUniform
%lstm_cell_15/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2'
%lstm_cell_15/dropout_3/GreaterEqual/yњ
#lstm_cell_15/dropout_3/GreaterEqualGreaterEqual<lstm_cell_15/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_15/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_15/dropout_3/GreaterEqualЌ
lstm_cell_15/dropout_3/CastCast'lstm_cell_15/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/CastЖ
lstm_cell_15/dropout_3/Mul_1Mullstm_cell_15/dropout_3/Mul:z:0lstm_cell_15/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/dropout_3/Mul_1~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0 lstm_cell_15/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0 lstm_cell_15/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0 lstm_cell_15/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_607683*
condR
while_cond_607682*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Зv
ы
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608172

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpZ
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
seed2тоЮ2&
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
seed2ЗУа2(
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
seed2Ж№2(
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
seed2Ѓ2(
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2,
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
ЮR
щ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_604544

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	*
readvariableop_resource:	 
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muld
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
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2,
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
цр

H__inference_sequential_6_layer_call_and_return_conditional_losses_606369

inputsE
2lstm_15_lstm_cell_15_split_readvariableop_resource:	C
4lstm_15_lstm_cell_15_split_1_readvariableop_resource:	?
,lstm_15_lstm_cell_15_readvariableop_resource:	 9
'dense_18_matmul_readvariableop_resource:  6
(dense_18_biasadd_readvariableop_resource: 9
'dense_19_matmul_readvariableop_resource: 6
(dense_19_biasadd_readvariableop_resource:
identityЂdense_18/BiasAdd/ReadVariableOpЂdense_18/MatMul/ReadVariableOpЂdense_19/BiasAdd/ReadVariableOpЂdense_19/MatMul/ReadVariableOpЂ/dense_19/bias/Regularizer/Square/ReadVariableOpЂ#lstm_15/lstm_cell_15/ReadVariableOpЂ%lstm_15/lstm_cell_15/ReadVariableOp_1Ђ%lstm_15/lstm_cell_15/ReadVariableOp_2Ђ%lstm_15/lstm_cell_15/ReadVariableOp_3Ђ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂ)lstm_15/lstm_cell_15/split/ReadVariableOpЂ+lstm_15/lstm_cell_15/split_1/ReadVariableOpЂlstm_15/whileT
lstm_15/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_15/Shape
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stack
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicel
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros/mul/y
lstm_15/zeros/mulMullstm_15/strided_slice:output:0lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/mulo
lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros/Less/y
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lessr
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros/packed/1Ѓ
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros/packedo
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros/Const
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/zerosp
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros_1/mul/y
lstm_15/zeros_1/mulMullstm_15/strided_slice:output:0lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/muls
lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros_1/Less/y
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessv
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/zeros_1/packed/1Љ
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros_1/packeds
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros_1/Const
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/zeros_1
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/perm
lstm_15/transpose	Transposeinputslstm_15/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stack
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_15/TensorArrayV2/element_shapeв
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2Я
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensor
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stack
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2Ќ
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_15/strided_slice_2
$lstm_15/lstm_cell_15/ones_like/ShapeShapelstm_15/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_15/lstm_cell_15/ones_like/Shape
$lstm_15/lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$lstm_15/lstm_cell_15/ones_like/Constи
lstm_15/lstm_cell_15/ones_likeFill-lstm_15/lstm_cell_15/ones_like/Shape:output:0-lstm_15/lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/ones_like
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dimЪ
)lstm_15/lstm_cell_15/split/ReadVariableOpReadVariableOp2lstm_15_lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02+
)lstm_15/lstm_cell_15/split/ReadVariableOpћ
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:01lstm_15/lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_15/lstm_cell_15/splitН
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMulС
lstm_15/lstm_cell_15/MatMul_1MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_1С
lstm_15/lstm_cell_15/MatMul_2MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_2С
lstm_15/lstm_cell_15/MatMul_3MatMul lstm_15/strided_slice_2:output:0#lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_3
&lstm_15/lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm_15/lstm_cell_15/split_1/split_dimЬ
+lstm_15/lstm_cell_15/split_1/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_15/lstm_cell_15/split_1/ReadVariableOpѓ
lstm_15/lstm_cell_15/split_1Split/lstm_15/lstm_cell_15/split_1/split_dim:output:03lstm_15/lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_15/lstm_cell_15/split_1Ч
lstm_15/lstm_cell_15/BiasAddBiasAdd%lstm_15/lstm_cell_15/MatMul:product:0%lstm_15/lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/BiasAddЭ
lstm_15/lstm_cell_15/BiasAdd_1BiasAdd'lstm_15/lstm_cell_15/MatMul_1:product:0%lstm_15/lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_1Э
lstm_15/lstm_cell_15/BiasAdd_2BiasAdd'lstm_15/lstm_cell_15/MatMul_2:product:0%lstm_15/lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_2Э
lstm_15/lstm_cell_15/BiasAdd_3BiasAdd'lstm_15/lstm_cell_15/MatMul_3:product:0%lstm_15/lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/BiasAdd_3Ў
lstm_15/lstm_cell_15/mulMullstm_15/zeros:output:0'lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mulВ
lstm_15/lstm_cell_15/mul_1Mullstm_15/zeros:output:0'lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_1В
lstm_15/lstm_cell_15/mul_2Mullstm_15/zeros:output:0'lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_2В
lstm_15/lstm_cell_15/mul_3Mullstm_15/zeros:output:0'lstm_15/lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_3И
#lstm_15/lstm_cell_15/ReadVariableOpReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02%
#lstm_15/lstm_cell_15/ReadVariableOpЅ
(lstm_15/lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_15/lstm_cell_15/strided_slice/stackЉ
*lstm_15/lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_15/lstm_cell_15/strided_slice/stack_1Љ
*lstm_15/lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_15/lstm_cell_15/strided_slice/stack_2њ
"lstm_15/lstm_cell_15/strided_sliceStridedSlice+lstm_15/lstm_cell_15/ReadVariableOp:value:01lstm_15/lstm_cell_15/strided_slice/stack:output:03lstm_15/lstm_cell_15/strided_slice/stack_1:output:03lstm_15/lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_15/lstm_cell_15/strided_sliceХ
lstm_15/lstm_cell_15/MatMul_4MatMullstm_15/lstm_cell_15/mul:z:0+lstm_15/lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_4П
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/BiasAdd:output:0'lstm_15/lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add
lstm_15/lstm_cell_15/SigmoidSigmoidlstm_15/lstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/SigmoidМ
%lstm_15/lstm_cell_15/ReadVariableOp_1ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_1Љ
*lstm_15/lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_15/lstm_cell_15/strided_slice_1/stack­
,lstm_15/lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,lstm_15/lstm_cell_15/strided_slice_1/stack_1­
,lstm_15/lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_1/stack_2
$lstm_15/lstm_cell_15/strided_slice_1StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_1:value:03lstm_15/lstm_cell_15/strided_slice_1/stack:output:05lstm_15/lstm_cell_15/strided_slice_1/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_1Щ
lstm_15/lstm_cell_15/MatMul_5MatMullstm_15/lstm_cell_15/mul_1:z:0-lstm_15/lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_5Х
lstm_15/lstm_cell_15/add_1AddV2'lstm_15/lstm_cell_15/BiasAdd_1:output:0'lstm_15/lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_1
lstm_15/lstm_cell_15/Sigmoid_1Sigmoidlstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/Sigmoid_1Џ
lstm_15/lstm_cell_15/mul_4Mul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_4М
%lstm_15/lstm_cell_15/ReadVariableOp_2ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_2Љ
*lstm_15/lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_15/lstm_cell_15/strided_slice_2/stack­
,lstm_15/lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2.
,lstm_15/lstm_cell_15/strided_slice_2/stack_1­
,lstm_15/lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_2/stack_2
$lstm_15/lstm_cell_15/strided_slice_2StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_2:value:03lstm_15/lstm_cell_15/strided_slice_2/stack:output:05lstm_15/lstm_cell_15/strided_slice_2/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_2Щ
lstm_15/lstm_cell_15/MatMul_6MatMullstm_15/lstm_cell_15/mul_2:z:0-lstm_15/lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_6Х
lstm_15/lstm_cell_15/add_2AddV2'lstm_15/lstm_cell_15/BiasAdd_2:output:0'lstm_15/lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_2
lstm_15/lstm_cell_15/ReluRelulstm_15/lstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/ReluМ
lstm_15/lstm_cell_15/mul_5Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_5Г
lstm_15/lstm_cell_15/add_3AddV2lstm_15/lstm_cell_15/mul_4:z:0lstm_15/lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_3М
%lstm_15/lstm_cell_15/ReadVariableOp_3ReadVariableOp,lstm_15_lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02'
%lstm_15/lstm_cell_15/ReadVariableOp_3Љ
*lstm_15/lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_15/lstm_cell_15/strided_slice_3/stack­
,lstm_15/lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_15/lstm_cell_15/strided_slice_3/stack_1­
,lstm_15/lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_15/lstm_cell_15/strided_slice_3/stack_2
$lstm_15/lstm_cell_15/strided_slice_3StridedSlice-lstm_15/lstm_cell_15/ReadVariableOp_3:value:03lstm_15/lstm_cell_15/strided_slice_3/stack:output:05lstm_15/lstm_cell_15/strided_slice_3/stack_1:output:05lstm_15/lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$lstm_15/lstm_cell_15/strided_slice_3Щ
lstm_15/lstm_cell_15/MatMul_7MatMullstm_15/lstm_cell_15/mul_3:z:0-lstm_15/lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/MatMul_7Х
lstm_15/lstm_cell_15/add_4AddV2'lstm_15/lstm_cell_15/BiasAdd_3:output:0'lstm_15/lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/add_4
lstm_15/lstm_cell_15/Sigmoid_2Sigmoidlstm_15/lstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_15/lstm_cell_15/Sigmoid_2
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/Relu_1Р
lstm_15/lstm_cell_15/mul_6Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_15/lstm_cell_15/mul_6
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2'
%lstm_15/TensorArrayV2_1/element_shapeи
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2_1^
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/time
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counterљ
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_15_lstm_cell_15_split_readvariableop_resource4lstm_15_lstm_cell_15_split_1_readvariableop_resource,lstm_15_lstm_cell_15_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_15_while_body_606208*%
condR
lstm_15_while_cond_606207*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_15/whileХ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStack
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_15/strided_slice_3/stack
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2Ъ
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_15/strided_slice_3
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/permХ
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_15/transpose_1v
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/runtimeЈ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_18/MatMul/ReadVariableOpЈ
dense_18/MatMulMatMul lstm_15/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/MatMulЇ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_18/BiasAdd/ReadVariableOpЅ
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_18/ReluЈ
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_19/MatMul/ReadVariableOpЃ
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19/MatMulЇ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOpЅ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19/BiasAddk
reshape_9/ShapeShapedense_19/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_9/Shape
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2в
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shapeЄ
reshape_9/ReshapeReshapedense_19/BiasAdd:output:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
reshape_9/Reshapeђ
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_15_lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/mulЧ
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/muly
IdentityIdentityreshape_9/Reshape:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2

IdentityЮ
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp0^dense_19/bias/Regularizer/Square/ReadVariableOp$^lstm_15/lstm_cell_15/ReadVariableOp&^lstm_15/lstm_cell_15/ReadVariableOp_1&^lstm_15/lstm_cell_15/ReadVariableOp_2&^lstm_15/lstm_cell_15/ReadVariableOp_3>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp*^lstm_15/lstm_cell_15/split/ReadVariableOp,^lstm_15/lstm_cell_15/split_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp2J
#lstm_15/lstm_cell_15/ReadVariableOp#lstm_15/lstm_cell_15/ReadVariableOp2N
%lstm_15/lstm_cell_15/ReadVariableOp_1%lstm_15/lstm_cell_15/ReadVariableOp_12N
%lstm_15/lstm_cell_15/ReadVariableOp_2%lstm_15/lstm_cell_15/ReadVariableOp_22N
%lstm_15/lstm_cell_15/ReadVariableOp_3%lstm_15/lstm_cell_15/ReadVariableOp_32~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2V
)lstm_15/lstm_cell_15/split/ReadVariableOp)lstm_15/lstm_cell_15/split/ReadVariableOp2Z
+lstm_15/lstm_cell_15/split_1/ReadVariableOp+lstm_15/lstm_cell_15/split_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
У
while_cond_604557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_604557___redundant_placeholder04
0while_while_cond_604557___redundant_placeholder14
0while_while_cond_604557___redundant_placeholder24
0while_while_cond_604557___redundant_placeholder3
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
Ђ
Љ
C__inference_lstm_15_layer_call_and_return_conditional_losses_606991
inputs_0=
*lstm_cell_15_split_readvariableop_resource:	;
,lstm_cell_15_split_1_readvariableop_resource:	7
$lstm_cell_15_readvariableop_resource:	 
identityЂ=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_15/ReadVariableOpЂlstm_cell_15/ReadVariableOp_1Ђlstm_cell_15/ReadVariableOp_2Ђlstm_cell_15/ReadVariableOp_3Ђ!lstm_cell_15/split/ReadVariableOpЂ#lstm_cell_15/split_1/ReadVariableOpЂwhileF
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
lstm_cell_15/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_15/ones_like/Shape
lstm_cell_15/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_15/ones_like/ConstИ
lstm_cell_15/ones_likeFill%lstm_cell_15/ones_like/Shape:output:0%lstm_cell_15/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/ones_like~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimВ
!lstm_cell_15/split/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02#
!lstm_cell_15/split/ReadVariableOpл
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0)lstm_cell_15/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_15/split
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0lstm_cell_15/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMulЁ
lstm_cell_15/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_15/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_1Ё
lstm_cell_15/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_15/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_2Ё
lstm_cell_15/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_15/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_3
lstm_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_15/split_1/split_dimД
#lstm_cell_15/split_1/ReadVariableOpReadVariableOp,lstm_cell_15_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_15/split_1/ReadVariableOpг
lstm_cell_15/split_1Split'lstm_cell_15/split_1/split_dim:output:0+lstm_cell_15/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_15/split_1Ї
lstm_cell_15/BiasAddBiasAddlstm_cell_15/MatMul:product:0lstm_cell_15/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd­
lstm_cell_15/BiasAdd_1BiasAddlstm_cell_15/MatMul_1:product:0lstm_cell_15/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_1­
lstm_cell_15/BiasAdd_2BiasAddlstm_cell_15/MatMul_2:product:0lstm_cell_15/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_2­
lstm_cell_15/BiasAdd_3BiasAddlstm_cell_15/MatMul_3:product:0lstm_cell_15/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/BiasAdd_3
lstm_cell_15/mulMulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul
lstm_cell_15/mul_1Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_1
lstm_cell_15/mul_2Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_2
lstm_cell_15/mul_3Mulzeros:output:0lstm_cell_15/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_3 
lstm_cell_15/ReadVariableOpReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp
 lstm_cell_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_15/strided_slice/stack
"lstm_cell_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice/stack_1
"lstm_cell_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_15/strided_slice/stack_2Ъ
lstm_cell_15/strided_sliceStridedSlice#lstm_cell_15/ReadVariableOp:value:0)lstm_cell_15/strided_slice/stack:output:0+lstm_cell_15/strided_slice/stack_1:output:0+lstm_cell_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_sliceЅ
lstm_cell_15/MatMul_4MatMullstm_cell_15/mul:z:0#lstm_cell_15/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_4
lstm_cell_15/addAddV2lstm_cell_15/BiasAdd:output:0lstm_cell_15/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add
lstm_cell_15/SigmoidSigmoidlstm_cell_15/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/SigmoidЄ
lstm_cell_15/ReadVariableOp_1ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_1
"lstm_cell_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_15/strided_slice_1/stack
$lstm_cell_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_15/strided_slice_1/stack_1
$lstm_cell_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_1/stack_2ж
lstm_cell_15/strided_slice_1StridedSlice%lstm_cell_15/ReadVariableOp_1:value:0+lstm_cell_15/strided_slice_1/stack:output:0-lstm_cell_15/strided_slice_1/stack_1:output:0-lstm_cell_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_1Љ
lstm_cell_15/MatMul_5MatMullstm_cell_15/mul_1:z:0%lstm_cell_15/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_5Ѕ
lstm_cell_15/add_1AddV2lstm_cell_15/BiasAdd_1:output:0lstm_cell_15/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mul_4Mullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_4Є
lstm_cell_15/ReadVariableOp_2ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_2
"lstm_cell_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_15/strided_slice_2/stack
$lstm_cell_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_15/strided_slice_2/stack_1
$lstm_cell_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_2/stack_2ж
lstm_cell_15/strided_slice_2StridedSlice%lstm_cell_15/ReadVariableOp_2:value:0+lstm_cell_15/strided_slice_2/stack:output:0-lstm_cell_15/strided_slice_2/stack_1:output:0-lstm_cell_15/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_2Љ
lstm_cell_15/MatMul_6MatMullstm_cell_15/mul_2:z:0%lstm_cell_15/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_6Ѕ
lstm_cell_15/add_2AddV2lstm_cell_15/BiasAdd_2:output:0lstm_cell_15/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_2x
lstm_cell_15/ReluRelulstm_cell_15/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu
lstm_cell_15/mul_5Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_5
lstm_cell_15/add_3AddV2lstm_cell_15/mul_4:z:0lstm_cell_15/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_3Є
lstm_cell_15/ReadVariableOp_3ReadVariableOp$lstm_cell_15_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_15/ReadVariableOp_3
"lstm_cell_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_15/strided_slice_3/stack
$lstm_cell_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_15/strided_slice_3/stack_1
$lstm_cell_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_15/strided_slice_3/stack_2ж
lstm_cell_15/strided_slice_3StridedSlice%lstm_cell_15/ReadVariableOp_3:value:0+lstm_cell_15/strided_slice_3/stack:output:0-lstm_cell_15/strided_slice_3/stack_1:output:0-lstm_cell_15/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_15/strided_slice_3Љ
lstm_cell_15/MatMul_7MatMullstm_cell_15/mul_3:z:0%lstm_cell_15/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/MatMul_7Ѕ
lstm_cell_15/add_4AddV2lstm_cell_15/BiasAdd_3:output:0lstm_cell_15/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/add_4
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_6Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_15/mul_6
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_15_split_readvariableop_resource,lstm_cell_15_split_1_readvariableop_resource$lstm_cell_15_readvariableop_resource*
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
while_body_606858*
condR
while_cond_606857*K
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
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_15_split_readvariableop_resource*
_output_shapes
:	*
dtype02?
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOpл
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareSquareElstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	20
.lstm_15/lstm_cell_15/kernel/Regularizer/SquareЏ
-lstm_15/lstm_cell_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm_15/lstm_cell_15/kernel/Regularizer/Constю
+lstm_15/lstm_cell_15/kernel/Regularizer/SumSum2lstm_15/lstm_cell_15/kernel/Regularizer/Square:y:06lstm_15/lstm_cell_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/SumЃ
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82/
-lstm_15/lstm_cell_15/kernel/Regularizer/mul/x№
+lstm_15/lstm_cell_15/kernel/Regularizer/mulMul6lstm_15/lstm_cell_15/kernel/Regularizer/mul/x:output:04lstm_15/lstm_cell_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+lstm_15/lstm_cell_15/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо
NoOpNoOp>^lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_15/ReadVariableOp^lstm_cell_15/ReadVariableOp_1^lstm_cell_15/ReadVariableOp_2^lstm_cell_15/ReadVariableOp_3"^lstm_cell_15/split/ReadVariableOp$^lstm_cell_15/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2~
=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp=lstm_15/lstm_cell_15/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_15/ReadVariableOplstm_cell_15/ReadVariableOp2>
lstm_cell_15/ReadVariableOp_1lstm_cell_15/ReadVariableOp_12>
lstm_cell_15/ReadVariableOp_2lstm_cell_15/ReadVariableOp_22>
lstm_cell_15/ReadVariableOp_3lstm_cell_15/ReadVariableOp_32F
!lstm_cell_15/split/ReadVariableOp!lstm_cell_15/split/ReadVariableOp2J
#lstm_cell_15/split_1/ReadVariableOp#lstm_cell_15/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
і
Љ
__inference_loss_fn_0_607972F
8dense_19_bias_regularizer_square_readvariableop_resource:
identityЂ/dense_19/bias/Regularizer/Square/ReadVariableOpз
/dense_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_19_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_19/bias/Regularizer/Square/ReadVariableOpЌ
 dense_19/bias/Regularizer/SquareSquare7dense_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_19/bias/Regularizer/Square
dense_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_19/bias/Regularizer/ConstЖ
dense_19/bias/Regularizer/SumSum$dense_19/bias/Regularizer/Square:y:0(dense_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/Sum
dense_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
dense_19/bias/Regularizer/mul/xИ
dense_19/bias/Regularizer/mulMul(dense_19/bias/Regularizer/mul/x:output:0&dense_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_19/bias/Regularizer/mulk
IdentityIdentity!dense_19/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense_19/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_19/bias/Regularizer/Square/ReadVariableOp/dense_19/bias/Regularizer/Square/ReadVariableOp"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
?
input_74
serving_default_input_7:0џџџџџџџџџA
	reshape_94
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
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
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"
_tf_keras_sequential
У
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_rnn_layer
Л

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
Л

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
Ѕ
trainable_variables
	variables
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
trainable_variables
)layer_metrics
	variables
*metrics

+layers
,non_trainable_variables
regularization_losses
-layer_regularization_losses
a__call__
b_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
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
trainable_variables
3layer_metrics
	variables
4metrics

5layers

6states
7non_trainable_variables
regularization_losses
8layer_regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_18/kernel
: 2dense_18/bias
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
trainable_variables
9layer_metrics
:metrics
	variables

;layers
<non_trainable_variables
regularization_losses
=layer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_19/kernel
:2dense_19/bias
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
trainable_variables
>layer_metrics
?metrics
	variables

@layers
Anon_trainable_variables
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
trainable_variables
Clayer_metrics
Dmetrics
	variables

Elayers
Fnon_trainable_variables
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
.:,	2lstm_15/lstm_cell_15/kernel
8:6	 2%lstm_15/lstm_cell_15/recurrent_kernel
(:&2lstm_15/lstm_cell_15/bias
 "
trackable_dict_wrapper
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
/trainable_variables
Ilayer_metrics
Jmetrics
0	variables

Klayers
Lnon_trainable_variables
1regularization_losses
Mlayer_regularization_losses
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
N
	Ntotal
	Ocount
P	variables
Q	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
&:$  2Adam/dense_18/kernel/m
 : 2Adam/dense_18/bias/m
&:$ 2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
3:1	2"Adam/lstm_15/lstm_cell_15/kernel/m
=:;	 2,Adam/lstm_15/lstm_cell_15/recurrent_kernel/m
-:+2 Adam/lstm_15/lstm_cell_15/bias/m
&:$  2Adam/dense_18/kernel/v
 : 2Adam/dense_18/bias/v
&:$ 2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
3:1	2"Adam/lstm_15/lstm_cell_15/kernel/v
=:;	 2,Adam/lstm_15/lstm_cell_15/recurrent_kernel/v
-:+2 Adam/lstm_15/lstm_cell_15/bias/v
ю2ы
H__inference_sequential_6_layer_call_and_return_conditional_losses_606369
H__inference_sequential_6_layer_call_and_return_conditional_losses_606704
H__inference_sequential_6_layer_call_and_return_conditional_losses_606025
H__inference_sequential_6_layer_call_and_return_conditional_losses_606059Р
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
-__inference_sequential_6_layer_call_fn_605545
-__inference_sequential_6_layer_call_fn_606723
-__inference_sequential_6_layer_call_fn_606742
-__inference_sequential_6_layer_call_fn_605991Р
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
!__inference__wrapped_model_604420input_7"
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
я2ь
C__inference_lstm_15_layer_call_and_return_conditional_losses_606991
C__inference_lstm_15_layer_call_and_return_conditional_losses_607298
C__inference_lstm_15_layer_call_and_return_conditional_losses_607541
C__inference_lstm_15_layer_call_and_return_conditional_losses_607848е
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
2
(__inference_lstm_15_layer_call_fn_607859
(__inference_lstm_15_layer_call_fn_607870
(__inference_lstm_15_layer_call_fn_607881
(__inference_lstm_15_layer_call_fn_607892е
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
D__inference_dense_18_layer_call_and_return_conditional_losses_607903Ђ
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
)__inference_dense_18_layer_call_fn_607912Ђ
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
D__inference_dense_19_layer_call_and_return_conditional_losses_607934Ђ
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
)__inference_dense_19_layer_call_fn_607943Ђ
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
E__inference_reshape_9_layer_call_and_return_conditional_losses_607956Ђ
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
*__inference_reshape_9_layer_call_fn_607961Ђ
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
Г2А
__inference_loss_fn_0_607972
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
ЫBШ
$__inference_signature_wrapper_606098input_7"
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
и2е
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608059
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608172О
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
Ђ2
-__inference_lstm_cell_15_layer_call_fn_608189
-__inference_lstm_cell_15_layer_call_fn_608206О
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
__inference_loss_fn_1_608217
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
!__inference__wrapped_model_604420z&('4Ђ1
*Ђ'
%"
input_7џџџџџџџџџ
Њ "9Њ6
4
	reshape_9'$
	reshape_9џџџџџџџџџЄ
D__inference_dense_18_layer_call_and_return_conditional_losses_607903\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
)__inference_dense_18_layer_call_fn_607912O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
D__inference_dense_19_layer_call_and_return_conditional_losses_607934\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_19_layer_call_fn_607943O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ;
__inference_loss_fn_0_607972Ђ

Ђ 
Њ " ;
__inference_loss_fn_1_608217&Ђ

Ђ 
Њ " Ф
C__inference_lstm_15_layer_call_and_return_conditional_losses_606991}&('OЂL
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
 Ф
C__inference_lstm_15_layer_call_and_return_conditional_losses_607298}&('OЂL
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
 Д
C__inference_lstm_15_layer_call_and_return_conditional_losses_607541m&('?Ђ<
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
 Д
C__inference_lstm_15_layer_call_and_return_conditional_losses_607848m&('?Ђ<
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
 
(__inference_lstm_15_layer_call_fn_607859p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
(__inference_lstm_15_layer_call_fn_607870p&('OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ 
(__inference_lstm_15_layer_call_fn_607881`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ 
(__inference_lstm_15_layer_call_fn_607892`&('?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ Ъ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608059§&('Ђ}
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
 Ъ
H__inference_lstm_cell_15_layer_call_and_return_conditional_losses_608172§&('Ђ}
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
 
-__inference_lstm_cell_15_layer_call_fn_608189э&('Ђ}
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
1/1џџџџџџџџџ 
-__inference_lstm_cell_15_layer_call_fn_608206э&('Ђ}
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
1/1џџџџџџџџџ Ѕ
E__inference_reshape_9_layer_call_and_return_conditional_losses_607956\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 }
*__inference_reshape_9_layer_call_fn_607961O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџО
H__inference_sequential_6_layer_call_and_return_conditional_losses_606025r&('<Ђ9
2Ђ/
%"
input_7џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 О
H__inference_sequential_6_layer_call_and_return_conditional_losses_606059r&('<Ђ9
2Ђ/
%"
input_7џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
H__inference_sequential_6_layer_call_and_return_conditional_losses_606369q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
H__inference_sequential_6_layer_call_and_return_conditional_losses_606704q&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 
-__inference_sequential_6_layer_call_fn_605545e&('<Ђ9
2Ђ/
%"
input_7џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_6_layer_call_fn_605991e&('<Ђ9
2Ђ/
%"
input_7џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
-__inference_sequential_6_layer_call_fn_606723d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_6_layer_call_fn_606742d&(';Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЎ
$__inference_signature_wrapper_606098&('?Ђ<
Ђ 
5Њ2
0
input_7%"
input_7џџџџџџџџџ"9Њ6
4
	reshape_9'$
	reshape_9џџџџџџџџџ