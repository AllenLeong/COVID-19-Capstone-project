ј√%
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ЋІ$
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:  *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
П
lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А**
shared_namelstm_4/lstm_cell_4/kernel
И
-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes
:	А*
dtype0
£
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel
Ь
7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel*
_output_shapes
:	 А*
dtype0
З
lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_4/lstm_cell_4/bias
А
+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
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
Ж
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
Э
 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m
Ц
4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes
:	А*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
™
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m
О
2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes	
:А*
dtype0
Ж
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
Э
 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v
Ц
4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes
:	А*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
™
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v
О
2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Ч,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*“+
value»+B≈+ BЊ+
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

)layers
*layer_metrics
trainable_variables
+metrics
,layer_regularization_losses
regularization_losses
-non_trainable_variables
	variables
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

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables
	variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
regularization_losses
=non_trainable_variables
	variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠

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
≠

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
VARIABLE_VALUElstm_4/lstm_cell_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_4/lstm_cell_4/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

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
≠

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
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
В
serving_default_input_3Placeholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3lstm_4/lstm_cell_4/kernellstm_4/lstm_cell_4/bias#lstm_4/lstm_cell_4/recurrent_kerneldense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
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
$__inference_signature_wrapper_102235
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpConst*)
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
__inference__traced_save_104426
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biastotalcountAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/m Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v*(
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
"__inference__traced_restore_104520≥™#
®

ѕ
lstm_4_while_cond_102344*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_102344___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_102344___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_102344___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_102344___redundant_placeholder3
lstm_4_while_identity
У
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
lstm_4/while/Lessr
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_4/while/Identity"7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
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
И
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_101680

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
а	
¶
-__inference_sequential_2_layer_call_fn_102848

inputs
unknown:	А
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_1016892
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_104069

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
’!
н
H__inference_sequential_2_layer_call_and_return_conditional_losses_102174
input_3 
lstm_4_102149:	А
lstm_4_102151:	А 
lstm_4_102153:	 А 
dense_6_102156:  
dense_6_102158:  
dense_7_102161: 
dense_7_102163:
identityИҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpЬ
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinput_3lstm_4_102149lstm_4_102151lstm_4_102153*
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1016262 
lstm_4/StatefulPartitionedCall∞
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_6_102156dense_6_102158*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1016452!
dense_6/StatefulPartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102161dense_7_102163*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1016612!
dense_7/StatefulPartitionedCallэ
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_1016802
reshape_3/PartitionedCall…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_102149*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_3/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityс
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
ѕР
Н
lstm_4_while_body_102345*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	АI
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АE
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:	 А
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	АG
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АC
0lstm_4_while_lstm_cell_4_readvariableop_resource:	 АИҐ'lstm_4/while/lstm_cell_4/ReadVariableOpҐ)lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ-lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp—
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype022
0lstm_4/while/TensorArrayV2Read/TensorListGetItemЮ
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/ones_like/ShapeЩ
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(lstm_4/while/lstm_cell_4/ones_like/Constи
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/ones_likeЦ
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dimЎ
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02/
-lstm_4/while/lstm_cell_4/split/ReadVariableOpЛ
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2 
lstm_4/while/lstm_cell_4/splitа
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_4/while/lstm_cell_4/MatMulд
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_1д
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_2д
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_3Ъ
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_4/while/lstm_cell_4/split_1/split_dimЏ
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpГ
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2"
 lstm_4/while/lstm_cell_4/split_1„
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/while/lstm_cell_4/BiasAddЁ
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_1Ё
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_2Ё
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_3Њ
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/mul¬
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_1¬
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_2¬
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_3∆
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02)
'lstm_4/while/lstm_cell_4/ReadVariableOp≠
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_4/while/lstm_cell_4/strided_slice/stack±
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_4/while/lstm_cell_4/strided_slice/stack_1±
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Т
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_4/while/lstm_cell_4/strided_slice’
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_4ѕ
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/add£
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/while/lstm_cell_4/Sigmoid 
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_1±
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_4/while/lstm_cell_4/strided_slice_1/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_1ў
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_5’
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_1©
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/Sigmoid_1љ
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_4 
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_2±
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_4/while/lstm_cell_4/strided_slice_2/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_2ў
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_6’
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_2Ь
lstm_4/while/lstm_cell_4/ReluRelu"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/Reluћ
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_5√
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_3 
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_3±
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_4/while/lstm_cell_4/strided_slice_3/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_3ў
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_7’
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_4©
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/Sigmoid_2†
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_4/while/lstm_cell_4/Relu_1–
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_6В
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype023
1lstm_4/while/TensorArrayV2Write/TensorListSetItemj
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add/yЕ
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/addn
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add_1/yЩ
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/add_1З
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity°
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_1Й
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_2ґ
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_3®
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/Identity_4®
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/Identity_5ш
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_4/while/NoOp"7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"ƒ
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 
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
•

ф
C__inference_dense_7_layer_call_and_return_conditional_losses_104047

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
’
√
while_cond_103807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103807___redundant_placeholder04
0while_while_cond_103807___redundant_placeholder14
0while_while_cond_103807___redundant_placeholder24
0while_while_cond_103807___redundant_placeholder3
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
Іх
—
!__inference__wrapped_model_100593
input_3P
=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource:	АN
?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource:	АJ
7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource:	 АE
3sequential_2_dense_6_matmul_readvariableop_resource:  B
4sequential_2_dense_6_biasadd_readvariableop_resource: E
3sequential_2_dense_7_matmul_readvariableop_resource: B
4sequential_2_dense_7_biasadd_readvariableop_resource:
identityИҐ+sequential_2/dense_6/BiasAdd/ReadVariableOpҐ*sequential_2/dense_6/MatMul/ReadVariableOpҐ+sequential_2/dense_7/BiasAdd/ReadVariableOpҐ*sequential_2/dense_7/MatMul/ReadVariableOpҐ.sequential_2/lstm_4/lstm_cell_4/ReadVariableOpҐ0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1Ґ0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2Ґ0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3Ґ4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOpҐ6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOpҐsequential_2/lstm_4/whilem
sequential_2/lstm_4/ShapeShapeinput_3*
T0*
_output_shapes
:2
sequential_2/lstm_4/ShapeЬ
'sequential_2/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/lstm_4/strided_slice/stack†
)sequential_2/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_4/strided_slice/stack_1†
)sequential_2/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_4/strided_slice/stack_2Џ
!sequential_2/lstm_4/strided_sliceStridedSlice"sequential_2/lstm_4/Shape:output:00sequential_2/lstm_4/strided_slice/stack:output:02sequential_2/lstm_4/strided_slice/stack_1:output:02sequential_2/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_2/lstm_4/strided_sliceД
sequential_2/lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_2/lstm_4/zeros/mul/yЉ
sequential_2/lstm_4/zeros/mulMul*sequential_2/lstm_4/strided_slice:output:0(sequential_2/lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_4/zeros/mulЗ
 sequential_2/lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 sequential_2/lstm_4/zeros/Less/yЈ
sequential_2/lstm_4/zeros/LessLess!sequential_2/lstm_4/zeros/mul:z:0)sequential_2/lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_2/lstm_4/zeros/LessК
"sequential_2/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_2/lstm_4/zeros/packed/1”
 sequential_2/lstm_4/zeros/packedPack*sequential_2/lstm_4/strided_slice:output:0+sequential_2/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_2/lstm_4/zeros/packedЗ
sequential_2/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_2/lstm_4/zeros/Const≈
sequential_2/lstm_4/zerosFill)sequential_2/lstm_4/zeros/packed:output:0(sequential_2/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_2/lstm_4/zerosИ
!sequential_2/lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_2/lstm_4/zeros_1/mul/y¬
sequential_2/lstm_4/zeros_1/mulMul*sequential_2/lstm_4/strided_slice:output:0*sequential_2/lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_4/zeros_1/mulЛ
"sequential_2/lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2$
"sequential_2/lstm_4/zeros_1/Less/yњ
 sequential_2/lstm_4/zeros_1/LessLess#sequential_2/lstm_4/zeros_1/mul:z:0+sequential_2/lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_2/lstm_4/zeros_1/LessО
$sequential_2/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_2/lstm_4/zeros_1/packed/1ў
"sequential_2/lstm_4/zeros_1/packedPack*sequential_2/lstm_4/strided_slice:output:0-sequential_2/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_2/lstm_4/zeros_1/packedЛ
!sequential_2/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_2/lstm_4/zeros_1/ConstЌ
sequential_2/lstm_4/zeros_1Fill+sequential_2/lstm_4/zeros_1/packed:output:0*sequential_2/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_2/lstm_4/zeros_1Э
"sequential_2/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_2/lstm_4/transpose/permЈ
sequential_2/lstm_4/transpose	Transposeinput_3+sequential_2/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
sequential_2/lstm_4/transposeЛ
sequential_2/lstm_4/Shape_1Shape!sequential_2/lstm_4/transpose:y:0*
T0*
_output_shapes
:2
sequential_2/lstm_4/Shape_1†
)sequential_2/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_4/strided_slice_1/stack§
+sequential_2/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_4/strided_slice_1/stack_1§
+sequential_2/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_4/strided_slice_1/stack_2ж
#sequential_2/lstm_4/strided_slice_1StridedSlice$sequential_2/lstm_4/Shape_1:output:02sequential_2/lstm_4/strided_slice_1/stack:output:04sequential_2/lstm_4/strided_slice_1/stack_1:output:04sequential_2/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_2/lstm_4/strided_slice_1≠
/sequential_2/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_2/lstm_4/TensorArrayV2/element_shapeВ
!sequential_2/lstm_4/TensorArrayV2TensorListReserve8sequential_2/lstm_4/TensorArrayV2/element_shape:output:0,sequential_2/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_2/lstm_4/TensorArrayV2з
Isequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Isequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape»
;sequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_2/lstm_4/transpose:y:0Rsequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor†
)sequential_2/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_4/strided_slice_2/stack§
+sequential_2/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_4/strided_slice_2/stack_1§
+sequential_2/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_4/strided_slice_2/stack_2ф
#sequential_2/lstm_4/strided_slice_2StridedSlice!sequential_2/lstm_4/transpose:y:02sequential_2/lstm_4/strided_slice_2/stack:output:04sequential_2/lstm_4/strided_slice_2/stack_1:output:04sequential_2/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2%
#sequential_2/lstm_4/strided_slice_2і
/sequential_2/lstm_4/lstm_cell_4/ones_like/ShapeShape"sequential_2/lstm_4/zeros:output:0*
T0*
_output_shapes
:21
/sequential_2/lstm_4/lstm_cell_4/ones_like/ShapeІ
/sequential_2/lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?21
/sequential_2/lstm_4/lstm_cell_4/ones_like/ConstД
)sequential_2/lstm_4/lstm_cell_4/ones_likeFill8sequential_2/lstm_4/lstm_cell_4/ones_like/Shape:output:08sequential_2/lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/ones_like§
/sequential_2/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/lstm_4/lstm_cell_4/split/split_dimл
4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype026
4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOpІ
%sequential_2/lstm_4/lstm_cell_4/splitSplit8sequential_2/lstm_4/lstm_cell_4/split/split_dim:output:0<sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2'
%sequential_2/lstm_4/lstm_cell_4/splitк
&sequential_2/lstm_4/lstm_cell_4/MatMulMatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_2/lstm_4/lstm_cell_4/MatMulо
(sequential_2/lstm_4/lstm_cell_4/MatMul_1MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_1о
(sequential_2/lstm_4/lstm_cell_4/MatMul_2MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_2о
(sequential_2/lstm_4/lstm_cell_4/MatMul_3MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_3®
1sequential_2/lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_2/lstm_4/lstm_cell_4/split_1/split_dimн
6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOpЯ
'sequential_2/lstm_4/lstm_cell_4/split_1Split:sequential_2/lstm_4/lstm_cell_4/split_1/split_dim:output:0>sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2)
'sequential_2/lstm_4/lstm_cell_4/split_1у
'sequential_2/lstm_4/lstm_cell_4/BiasAddBiasAdd0sequential_2/lstm_4/lstm_cell_4/MatMul:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_2/lstm_4/lstm_cell_4/BiasAddщ
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_1BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_1:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_1щ
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_2BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_2:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_2щ
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_3BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_3:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_3џ
#sequential_2/lstm_4/lstm_cell_4/mulMul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential_2/lstm_4/lstm_cell_4/mulя
%sequential_2/lstm_4/lstm_cell_4/mul_1Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_1я
%sequential_2/lstm_4/lstm_cell_4/mul_2Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_2я
%sequential_2/lstm_4/lstm_cell_4/mul_3Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_3ў
.sequential_2/lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype020
.sequential_2/lstm_4/lstm_cell_4/ReadVariableOpї
3sequential_2/lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential_2/lstm_4/lstm_cell_4/strided_slice/stackњ
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_1њ
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_2Љ
-sequential_2/lstm_4/lstm_cell_4/strided_sliceStridedSlice6sequential_2/lstm_4/lstm_cell_4/ReadVariableOp:value:0<sequential_2/lstm_4/lstm_cell_4/strided_slice/stack:output:0>sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_1:output:0>sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2/
-sequential_2/lstm_4/lstm_cell_4/strided_sliceс
(sequential_2/lstm_4/lstm_cell_4/MatMul_4MatMul'sequential_2/lstm_4/lstm_cell_4/mul:z:06sequential_2/lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_4л
#sequential_2/lstm_4/lstm_cell_4/addAddV20sequential_2/lstm_4/lstm_cell_4/BiasAdd:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential_2/lstm_4/lstm_cell_4/addЄ
'sequential_2/lstm_4/lstm_cell_4/SigmoidSigmoid'sequential_2/lstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_2/lstm_4/lstm_cell_4/SigmoidЁ
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype022
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1њ
5sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_1√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_2»
/sequential_2/lstm_4/lstm_cell_4/strided_slice_1StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_2/lstm_4/lstm_cell_4/strided_slice_1х
(sequential_2/lstm_4/lstm_cell_4/MatMul_5MatMul)sequential_2/lstm_4/lstm_cell_4/mul_1:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_5с
%sequential_2/lstm_4/lstm_cell_4/add_1AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_1:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/add_1Њ
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid)sequential_2/lstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_1№
%sequential_2/lstm_4/lstm_cell_4/mul_4Mul-sequential_2/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_2/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_4Ё
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype022
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2њ
5sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_1√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_2»
/sequential_2/lstm_4/lstm_cell_4/strided_slice_2StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_2/lstm_4/lstm_cell_4/strided_slice_2х
(sequential_2/lstm_4/lstm_cell_4/MatMul_6MatMul)sequential_2/lstm_4/lstm_cell_4/mul_2:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_6с
%sequential_2/lstm_4/lstm_cell_4/add_2AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_2:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/add_2±
$sequential_2/lstm_4/lstm_cell_4/ReluRelu)sequential_2/lstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_2/lstm_4/lstm_cell_4/Reluи
%sequential_2/lstm_4/lstm_cell_4/mul_5Mul+sequential_2/lstm_4/lstm_cell_4/Sigmoid:y:02sequential_2/lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_5я
%sequential_2/lstm_4/lstm_cell_4/add_3AddV2)sequential_2/lstm_4/lstm_cell_4/mul_4:z:0)sequential_2/lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/add_3Ё
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype022
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3њ
5sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   27
5sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_1√
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_2»
/sequential_2/lstm_4/lstm_cell_4/strided_slice_3StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask21
/sequential_2/lstm_4/lstm_cell_4/strided_slice_3х
(sequential_2/lstm_4/lstm_cell_4/MatMul_7MatMul)sequential_2/lstm_4/lstm_cell_4/mul_3:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(sequential_2/lstm_4/lstm_cell_4/MatMul_7с
%sequential_2/lstm_4/lstm_cell_4/add_4AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_3:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/add_4Њ
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid)sequential_2/lstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_2µ
&sequential_2/lstm_4/lstm_cell_4/Relu_1Relu)sequential_2/lstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_2/lstm_4/lstm_cell_4/Relu_1м
%sequential_2/lstm_4/lstm_cell_4/mul_6Mul-sequential_2/lstm_4/lstm_cell_4/Sigmoid_2:y:04sequential_2/lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/lstm_4/lstm_cell_4/mul_6Ј
1sequential_2/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    23
1sequential_2/lstm_4/TensorArrayV2_1/element_shapeИ
#sequential_2/lstm_4/TensorArrayV2_1TensorListReserve:sequential_2/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_2/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_2/lstm_4/TensorArrayV2_1v
sequential_2/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/lstm_4/timeІ
,sequential_2/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,sequential_2/lstm_4/while/maximum_iterationsТ
&sequential_2/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_2/lstm_4/while/loop_counter™
sequential_2/lstm_4/whileWhile/sequential_2/lstm_4/while/loop_counter:output:05sequential_2/lstm_4/while/maximum_iterations:output:0!sequential_2/lstm_4/time:output:0,sequential_2/lstm_4/TensorArrayV2_1:handle:0"sequential_2/lstm_4/zeros:output:0$sequential_2/lstm_4/zeros_1:output:0,sequential_2/lstm_4/strided_slice_1:output:0Ksequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_2_lstm_4_while_body_100444*1
cond)R'
%sequential_2_lstm_4_while_cond_100443*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
sequential_2/lstm_4/whileЁ
Dsequential_2/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2F
Dsequential_2/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeЄ
6sequential_2/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_2/lstm_4/while:output:3Msequential_2/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype028
6sequential_2/lstm_4/TensorArrayV2Stack/TensorListStack©
)sequential_2/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2+
)sequential_2/lstm_4/strided_slice_3/stack§
+sequential_2/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_2/lstm_4/strided_slice_3/stack_1§
+sequential_2/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_4/strided_slice_3/stack_2Т
#sequential_2/lstm_4/strided_slice_3StridedSlice?sequential_2/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_2/lstm_4/strided_slice_3/stack:output:04sequential_2/lstm_4/strided_slice_3/stack_1:output:04sequential_2/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2%
#sequential_2/lstm_4/strided_slice_3°
$sequential_2/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_2/lstm_4/transpose_1/permх
sequential_2/lstm_4/transpose_1	Transpose?sequential_2/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_2/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2!
sequential_2/lstm_4/transpose_1О
sequential_2/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_2/lstm_4/runtimeћ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOpЎ
sequential_2/dense_6/MatMulMatMul,sequential_2/lstm_4/strided_slice_3:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_2/dense_6/MatMulЋ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOp’
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_2/dense_6/BiasAddЧ
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_2/dense_6/Reluћ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_2/dense_7/MatMul/ReadVariableOp”
sequential_2/dense_7/MatMulMatMul'sequential_2/dense_6/Relu:activations:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_2/dense_7/MatMulЋ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_7/BiasAdd/ReadVariableOp’
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_2/dense_7/BiasAddС
sequential_2/reshape_3/ShapeShape%sequential_2/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_2/reshape_3/ShapeҐ
*sequential_2/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_2/reshape_3/strided_slice/stack¶
,sequential_2/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_3/strided_slice/stack_1¶
,sequential_2/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_3/strided_slice/stack_2м
$sequential_2/reshape_3/strided_sliceStridedSlice%sequential_2/reshape_3/Shape:output:03sequential_2/reshape_3/strided_slice/stack:output:05sequential_2/reshape_3/strided_slice/stack_1:output:05sequential_2/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_2/reshape_3/strided_sliceТ
&sequential_2/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_3/Reshape/shape/1Т
&sequential_2/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_3/Reshape/shape/2У
$sequential_2/reshape_3/Reshape/shapePack-sequential_2/reshape_3/strided_slice:output:0/sequential_2/reshape_3/Reshape/shape/1:output:0/sequential_2/reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/reshape_3/Reshape/shape„
sequential_2/reshape_3/ReshapeReshape%sequential_2/dense_7/BiasAdd:output:0-sequential_2/reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 
sequential_2/reshape_3/ReshapeЖ
IdentityIdentity'sequential_2/reshape_3/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЏ
NoOpNoOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp/^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp1^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_11^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_21^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_35^sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp7^sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp^sequential_2/lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp2`
.sequential_2/lstm_4/lstm_cell_4/ReadVariableOp.sequential_2/lstm_4/lstm_cell_4/ReadVariableOp2d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_10sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_12d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_20sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_22d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_30sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_32l
4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp2p
6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp26
sequential_2/lstm_4/whilesequential_2/lstm_4/while:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
РB
ƒ
__inference__traced_save_104426
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesЊ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
…: :  : : :: : : : : :	А:	 А:А: : :  : : ::	А:	 А:А:  : : ::	А:	 А:А: 2(
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
:	А:%!

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
:	А:%!

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
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:

_output_shapes
: 
г	
І
-__inference_sequential_2_layer_call_fn_102146
input_3
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_1021102
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
Б
ф
C__inference_dense_6_layer_call_and_return_conditional_losses_104028

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
ƒ~
Ш	
while_body_102983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
њ%
№
while_body_101028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_101052_0:	А)
while_lstm_cell_4_101054_0:	А-
while_lstm_cell_4_101056_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_101052:	А'
while_lstm_cell_4_101054:	А+
while_lstm_cell_4_101056:	 АИҐ)while/lstm_cell_4/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_101052_0while_lstm_cell_4_101054_0while_lstm_cell_4_101056_0*
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1009502+
)while/lstm_cell_4/StatefulPartitionedCallц
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/Identity_3£
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
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
while_lstm_cell_4_101052while_lstm_cell_4_101052_0"6
while_lstm_cell_4_101054while_lstm_cell_4_101054_0"6
while_lstm_cell_4_101056while_lstm_cell_4_101056_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
Ч≤
‘
%sequential_2_lstm_4_while_body_100444D
@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counterJ
Fsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations)
%sequential_2_lstm_4_while_placeholder+
'sequential_2_lstm_4_while_placeholder_1+
'sequential_2_lstm_4_while_placeholder_2+
'sequential_2_lstm_4_while_placeholder_3C
?sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1_0
{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	АV
Gsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АR
?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0:	 А&
"sequential_2_lstm_4_while_identity(
$sequential_2_lstm_4_while_identity_1(
$sequential_2_lstm_4_while_identity_2(
$sequential_2_lstm_4_while_identity_3(
$sequential_2_lstm_4_while_identity_4(
$sequential_2_lstm_4_while_identity_5A
=sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1}
ysequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensorV
Csequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource:	АT
Esequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АP
=sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource:	 АИҐ4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOpҐ6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpл
Ksequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2M
Ksequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeЋ
=sequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_2_lstm_4_while_placeholderTsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02?
=sequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem≈
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/ShapeShape'sequential_2_lstm_4_while_placeholder_2*
T0*
_output_shapes
:27
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/Shape≥
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?27
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/ConstЬ
/sequential_2/lstm_4/while/lstm_cell_4/ones_likeFill>sequential_2/lstm_4/while/lstm_cell_4/ones_like/Shape:output:0>sequential_2/lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/ones_like∞
5sequential_2/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_2/lstm_4/while/lstm_cell_4/split/split_dim€
:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOpEsequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02<
:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOpњ
+sequential_2/lstm_4/while/lstm_cell_4/splitSplit>sequential_2/lstm_4/while/lstm_cell_4/split/split_dim:output:0Bsequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2-
+sequential_2/lstm_4/while/lstm_cell_4/splitФ
,sequential_2/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_2/lstm_4/while/lstm_cell_4/MatMulШ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_1MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_1Ш
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_2MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_2Ш
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_3MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_3і
7sequential_2/lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_2/lstm_4/while/lstm_cell_4/split_1/split_dimБ
<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOpGsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpЈ
-sequential_2/lstm_4/while/lstm_cell_4/split_1Split@sequential_2/lstm_4/while/lstm_cell_4/split_1/split_dim:output:0Dsequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2/
-sequential_2/lstm_4/while/lstm_cell_4/split_1Л
-sequential_2/lstm_4/while/lstm_cell_4/BiasAddBiasAdd6sequential_2/lstm_4/while/lstm_cell_4/MatMul:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_2/lstm_4/while/lstm_cell_4/BiasAddС
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_1:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_1С
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_2:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_2С
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_3:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_3т
)sequential_2/lstm_4/while/lstm_cell_4/mulMul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/while/lstm_cell_4/mulц
+sequential_2/lstm_4/while/lstm_cell_4/mul_1Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_1ц
+sequential_2/lstm_4/while/lstm_cell_4/mul_2Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_2ц
+sequential_2/lstm_4/while/lstm_cell_4/mul_3Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_3н
4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype026
4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp«
9sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stackЋ
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_1Ћ
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_2а
3sequential_2/lstm_4/while/lstm_cell_4/strided_sliceStridedSlice<sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp:value:0Bsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack:output:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask25
3sequential_2/lstm_4/while/lstm_cell_4/strided_sliceЙ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_4MatMul-sequential_2/lstm_4/while/lstm_cell_4/mul:z:0<sequential_2/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_4Г
)sequential_2/lstm_4/while/lstm_cell_4/addAddV26sequential_2/lstm_4/while/lstm_cell_4/BiasAdd:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_2/lstm_4/while/lstm_cell_4/add 
-sequential_2/lstm_4/while/lstm_cell_4/SigmoidSigmoid-sequential_2/lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_2/lstm_4/while/lstm_cell_4/Sigmoidс
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype028
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1Ћ
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2=
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stackѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1ѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2м
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1Н
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_5MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_1:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_5Й
+sequential_2/lstm_4/while/lstm_cell_4/add_1AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_1:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/add_1–
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid/sequential_2/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_1с
+sequential_2/lstm_4/while/lstm_cell_4/mul_4Mul3sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_2_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_4с
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype028
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2Ћ
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2=
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stackѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1ѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2м
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2Н
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_6MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_2:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_6Й
+sequential_2/lstm_4/while/lstm_cell_4/add_2AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_2:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/add_2√
*sequential_2/lstm_4/while/lstm_cell_4/ReluRelu/sequential_2/lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_2/lstm_4/while/lstm_cell_4/ReluА
+sequential_2/lstm_4/while/lstm_cell_4/mul_5Mul1sequential_2/lstm_4/while/lstm_cell_4/Sigmoid:y:08sequential_2/lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_5ч
+sequential_2/lstm_4/while/lstm_cell_4/add_3AddV2/sequential_2/lstm_4/while/lstm_cell_4/mul_4:z:0/sequential_2/lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/add_3с
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype028
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3Ћ
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2=
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stackѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1ѕ
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2м
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask27
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3Н
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_7MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_3:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 20
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_7Й
+sequential_2/lstm_4/while/lstm_cell_4/add_4AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_3:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/add_4–
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid/sequential_2/lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_2«
,sequential_2/lstm_4/while/lstm_cell_4/Relu_1Relu/sequential_2/lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_2/lstm_4/while/lstm_cell_4/Relu_1Д
+sequential_2/lstm_4/while/lstm_cell_4/mul_6Mul3sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_2:y:0:sequential_2/lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_2/lstm_4/while/lstm_cell_4/mul_6√
>sequential_2/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_2_lstm_4_while_placeholder_1%sequential_2_lstm_4_while_placeholder/sequential_2/lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype02@
>sequential_2/lstm_4/while/TensorArrayV2Write/TensorListSetItemД
sequential_2/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_2/lstm_4/while/add/yє
sequential_2/lstm_4/while/addAddV2%sequential_2_lstm_4_while_placeholder(sequential_2/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_4/while/addИ
!sequential_2/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_2/lstm_4/while/add_1/yЏ
sequential_2/lstm_4/while/add_1AddV2@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counter*sequential_2/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_4/while/add_1ї
"sequential_2/lstm_4/while/IdentityIdentity#sequential_2/lstm_4/while/add_1:z:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_2/lstm_4/while/Identityв
$sequential_2/lstm_4/while/Identity_1IdentityFsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_4/while/Identity_1љ
$sequential_2/lstm_4/while/Identity_2Identity!sequential_2/lstm_4/while/add:z:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_4/while/Identity_2к
$sequential_2/lstm_4/while/Identity_3IdentityNsequential_2/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_4/while/Identity_3№
$sequential_2/lstm_4/while/Identity_4Identity/sequential_2/lstm_4/while/lstm_cell_4/mul_6:z:0^sequential_2/lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_2/lstm_4/while/Identity_4№
$sequential_2/lstm_4/while/Identity_5Identity/sequential_2/lstm_4/while/lstm_cell_4/add_3:z:0^sequential_2/lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_2/lstm_4/while/Identity_5а
sequential_2/lstm_4/while/NoOpNoOp5^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp7^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_17^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_27^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3;^sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp=^sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_2/lstm_4/while/NoOp"Q
"sequential_2_lstm_4_while_identity+sequential_2/lstm_4/while/Identity:output:0"U
$sequential_2_lstm_4_while_identity_1-sequential_2/lstm_4/while/Identity_1:output:0"U
$sequential_2_lstm_4_while_identity_2-sequential_2/lstm_4/while/Identity_2:output:0"U
$sequential_2_lstm_4_while_identity_3-sequential_2/lstm_4/while/Identity_3:output:0"U
$sequential_2_lstm_4_while_identity_4-sequential_2/lstm_4/while/Identity_4:output:0"U
$sequential_2_lstm_4_while_identity_5-sequential_2/lstm_4/while/Identity_5:output:0"А
=sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0"Р
Esequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resourceGsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"М
Csequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resourceEsequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"А
=sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1?sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1_0"ш
ysequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2l
4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp2p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_16sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_12p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_26sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_22p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_36sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_32x
:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp2|
<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 
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
“!
м
H__inference_sequential_2_layer_call_and_return_conditional_losses_102110

inputs 
lstm_4_102085:	А
lstm_4_102087:	А 
lstm_4_102089:	 А 
dense_6_102092:  
dense_6_102094:  
dense_7_102097: 
dense_7_102099:
identityИҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpЫ
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_102085lstm_4_102087lstm_4_102089*
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1020522 
lstm_4/StatefulPartitionedCall∞
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_6_102092dense_6_102094*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1016452!
dense_6/StatefulPartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102097dense_7_102099*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1016612!
dense_7/StatefulPartitionedCallэ
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_1016802
reshape_3/PartitionedCall…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_102085*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_3/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityс
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥	
Ю
$__inference_signature_wrapper_102235
input_3
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
!__inference__wrapped_model_1005932
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
“!
м
H__inference_sequential_2_layer_call_and_return_conditional_losses_101689

inputs 
lstm_4_101627:	А
lstm_4_101629:	А 
lstm_4_101631:	 А 
dense_6_101646:  
dense_6_101648:  
dense_7_101662: 
dense_7_101664:
identityИҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpЫ
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_101627lstm_4_101629lstm_4_101631*
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1016262 
lstm_4/StatefulPartitionedCall∞
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_6_101646dense_6_101648*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1016452!
dense_6/StatefulPartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_101662dense_7_101664*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1016612!
dense_7/StatefulPartitionedCallэ
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_1016802
reshape_3/PartitionedCall…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_101627*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_3/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityс
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
п
Х
(__inference_dense_6_layer_call_fn_104037

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallу
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1016452
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
п
Х
(__inference_dense_7_layer_call_fn_104056

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallу
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1016612
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
©Ќ
Э
B__inference_lstm_4_layer_call_and_return_conditional_losses_103423
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout/Constѓ
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeч
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2№њ©22
0lstm_cell_4/dropout/random_uniform/RandomUniformН
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell_4/dropout/GreaterEqual/yо
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_1/Constµ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeэ
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Эж†24
2lstm_cell_4/dropout_1/random_uniform/RandomUniformС
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_1/GreaterEqual/yц
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_2/Constµ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeэ
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2≥шМ24
2lstm_cell_4/dropout_2/random_uniform/RandomUniformС
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_2/GreaterEqual/yц
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_3/Constµ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2УїЄ24
2lstm_cell_4/dropout_3/random_uniform/RandomUniformС
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_3/GreaterEqual/yц
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_103258*
condR
while_cond_103257*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Г»
Н
lstm_4_while_body_102642*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	АI
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АE
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:	 А
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	АG
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АC
0lstm_4_while_lstm_cell_4_readvariableop_resource:	 АИҐ'lstm_4/while/lstm_cell_4/ReadVariableOpҐ)lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ-lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp—
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype022
0lstm_4/while/TensorArrayV2Read/TensorListGetItemЮ
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/ones_like/ShapeЩ
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(lstm_4/while/lstm_cell_4/ones_like/Constи
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/ones_likeХ
&lstm_4/while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2(
&lstm_4/while/lstm_cell_4/dropout/Constг
$lstm_4/while/lstm_cell_4/dropout/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:0/lstm_4/while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$lstm_4/while/lstm_cell_4/dropout/MulЂ
&lstm_4/while/lstm_cell_4/dropout/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_4/while/lstm_cell_4/dropout/ShapeЮ
=lstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform/lstm_4/while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2°ЭЪ2?
=lstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniformІ
/lstm_4/while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>21
/lstm_4/while/lstm_cell_4/dropout/GreaterEqual/yҐ
-lstm_4/while/lstm_cell_4/dropout/GreaterEqualGreaterEqualFlstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:08lstm_4/while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-lstm_4/while/lstm_cell_4/dropout/GreaterEqual 
%lstm_4/while/lstm_cell_4/dropout/CastCast1lstm_4/while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2'
%lstm_4/while/lstm_cell_4/dropout/Castё
&lstm_4/while/lstm_cell_4/dropout/Mul_1Mul(lstm_4/while/lstm_cell_4/dropout/Mul:z:0)lstm_4/while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_4/while/lstm_cell_4/dropout/Mul_1Щ
(lstm_4/while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2*
(lstm_4/while/lstm_cell_4/dropout_1/Constй
&lstm_4/while/lstm_cell_4/dropout_1/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_4/while/lstm_cell_4/dropout_1/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_1/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_1/Shape§
?lstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЫЫ≈2A
?lstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformЂ
1lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>23
1lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/y™
/lstm_4/while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_1/CastCast3lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2)
'lstm_4/while/lstm_cell_4/dropout_1/Castж
(lstm_4/while/lstm_cell_4/dropout_1/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_1/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_4/while/lstm_cell_4/dropout_1/Mul_1Щ
(lstm_4/while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2*
(lstm_4/while/lstm_cell_4/dropout_2/Constй
&lstm_4/while/lstm_cell_4/dropout_2/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_4/while/lstm_cell_4/dropout_2/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_2/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_2/Shape§
?lstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ШкЋ2A
?lstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformЂ
1lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>23
1lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/y™
/lstm_4/while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_2/CastCast3lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2)
'lstm_4/while/lstm_cell_4/dropout_2/Castж
(lstm_4/while/lstm_cell_4/dropout_2/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_2/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_4/while/lstm_cell_4/dropout_2/Mul_1Щ
(lstm_4/while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2*
(lstm_4/while/lstm_cell_4/dropout_3/Constй
&lstm_4/while/lstm_cell_4/dropout_3/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&lstm_4/while/lstm_cell_4/dropout_3/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_3/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_3/Shape§
?lstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2зти2A
?lstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformЂ
1lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>23
1lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/y™
/lstm_4/while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 21
/lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_3/CastCast3lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2)
'lstm_4/while/lstm_cell_4/dropout_3/Castж
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_3/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Ц
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dimЎ
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02/
-lstm_4/while/lstm_cell_4/split/ReadVariableOpЛ
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2 
lstm_4/while/lstm_cell_4/splitа
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_4/while/lstm_cell_4/MatMulд
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_1д
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_2д
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_3Ъ
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_4/while/lstm_cell_4/split_1/split_dimЏ
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpГ
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2"
 lstm_4/while/lstm_cell_4/split_1„
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/while/lstm_cell_4/BiasAddЁ
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_1Ё
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_2Ё
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/BiasAdd_3љ
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2*lstm_4/while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/mul√
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_1√
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_2√
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_3∆
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02)
'lstm_4/while/lstm_cell_4/ReadVariableOp≠
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_4/while/lstm_cell_4/strided_slice/stack±
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_4/while/lstm_cell_4/strided_slice/stack_1±
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Т
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2(
&lstm_4/while/lstm_cell_4/strided_slice’
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_4ѕ
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/add£
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/while/lstm_cell_4/Sigmoid 
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_1±
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_4/while/lstm_cell_4/strided_slice_1/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   22
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_1ў
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_5’
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_1©
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/Sigmoid_1љ
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_4 
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_2±
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_4/while/lstm_cell_4/strided_slice_2/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   22
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_2ў
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_6’
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_2Ь
lstm_4/while/lstm_cell_4/ReluRelu"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/lstm_cell_4/Reluћ
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_5√
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_3 
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_3±
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   20
.lstm_4/while/lstm_cell_4/strided_slice_3/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1µ
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Ю
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_3ў
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/while/lstm_cell_4/MatMul_7’
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/add_4©
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/while/lstm_cell_4/Sigmoid_2†
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_4/while/lstm_cell_4/Relu_1–
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/while/lstm_cell_4/mul_6В
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype023
1lstm_4/while/TensorArrayV2Write/TensorListSetItemj
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add/yЕ
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/addn
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add_1/yЩ
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/add_1З
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity°
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_1Й
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_2ґ
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 2
lstm_4/while/Identity_3®
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/Identity_4®
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/while/Identity_5ш
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_4/while/NoOp"7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"ƒ
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 
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
while_cond_101027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101027___redundant_placeholder04
0while_while_cond_101027___redundant_placeholder14
0while_while_cond_101027___redundant_placeholder24
0while_while_cond_101027___redundant_placeholder3
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
е{
в
"__inference__traced_restore_104520
file_prefix1
assignvariableop_dense_6_kernel:  -
assignvariableop_1_dense_6_bias: 3
!assignvariableop_2_dense_7_kernel: -
assignvariableop_3_dense_7_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
,assignvariableop_9_lstm_4_lstm_cell_4_kernel:	АJ
7assignvariableop_10_lstm_4_lstm_cell_4_recurrent_kernel:	 А:
+assignvariableop_11_lstm_4_lstm_cell_4_bias:	А#
assignvariableop_12_total: #
assignvariableop_13_count: ;
)assignvariableop_14_adam_dense_6_kernel_m:  5
'assignvariableop_15_adam_dense_6_bias_m: ;
)assignvariableop_16_adam_dense_7_kernel_m: 5
'assignvariableop_17_adam_dense_7_bias_m:G
4assignvariableop_18_adam_lstm_4_lstm_cell_4_kernel_m:	АQ
>assignvariableop_19_adam_lstm_4_lstm_cell_4_recurrent_kernel_m:	 АA
2assignvariableop_20_adam_lstm_4_lstm_cell_4_bias_m:	А;
)assignvariableop_21_adam_dense_6_kernel_v:  5
'assignvariableop_22_adam_dense_6_bias_v: ;
)assignvariableop_23_adam_dense_7_kernel_v: 5
'assignvariableop_24_adam_dense_7_bias_v:G
4assignvariableop_25_adam_lstm_4_lstm_cell_4_kernel_v:	АQ
>assignvariableop_26_adam_lstm_4_lstm_cell_4_recurrent_kernel_v:	 АA
2assignvariableop_27_adam_lstm_4_lstm_cell_4_bias_v:	А
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

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_4_lstm_cell_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10њ
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_4_lstm_cell_4_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11≥
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_4_lstm_cell_4_biasIdentity_11:output:0"/device:CPU:0*
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
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_6_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ѓ
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_6_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_7_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ѓ
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_7_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Љ
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_4_lstm_cell_4_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19∆
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ї
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_lstm_4_lstm_cell_4_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21±
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_6_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ѓ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_6_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_7_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ѓ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_7_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Љ
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_4_lstm_cell_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26∆
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ї
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_lstm_4_lstm_cell_4_bias_vIdentity_27:output:0"/device:CPU:0*
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
’
√
while_cond_103532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103532___redundant_placeholder04
0while_while_cond_103532___redundant_placeholder14
0while_while_cond_103532___redundant_placeholder24
0while_while_cond_103532___redundant_placeholder3
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
ї∞
Ш	
while_body_103258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeЗ
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell_4/dropout/Const«
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeИ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЕрB28
6while/lstm_cell_4/dropout/random_uniform/RandomUniformЩ
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell_4/dropout/GreaterEqual/yЖ
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout/Mul_1Л
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_1/ConstЌ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Ё√Э2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_1/GreaterEqual/yО
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_1/Mul_1Л
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_2/ConstЌ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeП
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2јЎх2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_2/GreaterEqual/yО
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_2/Mul_1Л
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_3/ConstЌ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeП
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЋЗ¬2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_3/GreaterEqual/yО
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
Љ
ґ
'__inference_lstm_4_layer_call_fn_103984
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallБ
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1008062
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Б
ф
C__inference_dense_6_layer_call_and_return_conditional_losses_101645

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
’
√
while_cond_100730
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100730___redundant_placeholder04
0while_while_cond_100730___redundant_placeholder14
0while_while_cond_100730___redundant_placeholder24
0while_while_cond_100730___redundant_placeholder3
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
§
і
'__inference_lstm_4_layer_call_fn_104017

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCall€
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1020522
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Н†
Э
B__inference_lstm_4_layer_call_and_return_conditional_losses_103116
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_102983*
condR
while_cond_102982*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
іД
Ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102829

inputsC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	АA
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	А=
*lstm_4_lstm_cell_4_readvariableop_resource:	 А8
&dense_6_matmul_readvariableop_resource:  5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identityИҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ!lstm_4/lstm_cell_4/ReadVariableOpҐ#lstm_4/lstm_cell_4/ReadVariableOp_1Ґ#lstm_4/lstm_cell_4/ReadVariableOp_2Ґ#lstm_4/lstm_cell_4/ReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ'lstm_4/lstm_cell_4/split/ReadVariableOpҐ)lstm_4/lstm_cell_4/split_1/ReadVariableOpҐlstm_4/whileR
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_4/ShapeВ
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stackЖ
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1Ж
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2М
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicej
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros/mul/yИ
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_4/zeros/Less/yГ
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessp
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros/packed/1Я
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/ConstС
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros_1/mul/yО
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_4/zeros_1/Less/yЛ
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lesst
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros_1/packed/1•
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/ConstЩ
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/zeros_1Г
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/permП
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_4/transposed
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:2
lstm_4/Shape_1Ж
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_1/stackК
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_1К
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_2Ш
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slice_1У
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_4/TensorArrayV2/element_shapeќ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2Ќ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_4/TensorArrayUnstack/TensorListFromTensorЖ
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_2/stackК
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_1К
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_2¶
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_4/strided_slice_2Н
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/ones_like/ShapeН
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"lstm_4/lstm_cell_4/ones_like/Const–
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/ones_likeЙ
 lstm_4/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2"
 lstm_4/lstm_cell_4/dropout/ConstЋ
lstm_4/lstm_cell_4/dropout/MulMul%lstm_4/lstm_cell_4/ones_like:output:0)lstm_4/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_4/lstm_cell_4/dropout/MulЩ
 lstm_4/lstm_cell_4/dropout/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_4/lstm_cell_4/dropout/ShapeМ
7lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform)lstm_4/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЕџЦ29
7lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniformЫ
)lstm_4/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2+
)lstm_4/lstm_cell_4/dropout/GreaterEqual/yК
'lstm_4/lstm_cell_4/dropout/GreaterEqualGreaterEqual@lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniform:output:02lstm_4/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'lstm_4/lstm_cell_4/dropout/GreaterEqualЄ
lstm_4/lstm_cell_4/dropout/CastCast+lstm_4/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2!
lstm_4/lstm_cell_4/dropout/Cast∆
 lstm_4/lstm_cell_4/dropout/Mul_1Mul"lstm_4/lstm_cell_4/dropout/Mul:z:0#lstm_4/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/lstm_cell_4/dropout/Mul_1Н
"lstm_4/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2$
"lstm_4/lstm_cell_4/dropout_1/Const—
 lstm_4/lstm_cell_4/dropout_1/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/lstm_cell_4/dropout_1/MulЭ
"lstm_4/lstm_cell_4/dropout_1/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_1/ShapeТ
9lstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2лѕх2;
9lstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniformЯ
+lstm_4/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2-
+lstm_4/lstm_cell_4/dropout_1/GreaterEqual/yТ
)lstm_4/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_4/lstm_cell_4/dropout_1/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_1/CastCast-lstm_4/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/lstm_cell_4/dropout_1/Castќ
"lstm_4/lstm_cell_4/dropout_1/Mul_1Mul$lstm_4/lstm_cell_4/dropout_1/Mul:z:0%lstm_4/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/lstm_cell_4/dropout_1/Mul_1Н
"lstm_4/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2$
"lstm_4/lstm_cell_4/dropout_2/Const—
 lstm_4/lstm_cell_4/dropout_2/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/lstm_cell_4/dropout_2/MulЭ
"lstm_4/lstm_cell_4/dropout_2/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_2/ShapeТ
9lstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ђќ∆2;
9lstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniformЯ
+lstm_4/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2-
+lstm_4/lstm_cell_4/dropout_2/GreaterEqual/yТ
)lstm_4/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_4/lstm_cell_4/dropout_2/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_2/CastCast-lstm_4/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/lstm_cell_4/dropout_2/Castќ
"lstm_4/lstm_cell_4/dropout_2/Mul_1Mul$lstm_4/lstm_cell_4/dropout_2/Mul:z:0%lstm_4/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/lstm_cell_4/dropout_2/Mul_1Н
"lstm_4/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2$
"lstm_4/lstm_cell_4/dropout_3/Const—
 lstm_4/lstm_cell_4/dropout_3/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_4/lstm_cell_4/dropout_3/MulЭ
"lstm_4/lstm_cell_4/dropout_3/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_3/ShapeС
9lstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ЇЎU2;
9lstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniformЯ
+lstm_4/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2-
+lstm_4/lstm_cell_4/dropout_3/GreaterEqual/yТ
)lstm_4/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)lstm_4/lstm_cell_4/dropout_3/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_3/CastCast-lstm_4/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!lstm_4/lstm_cell_4/dropout_3/Castќ
"lstm_4/lstm_cell_4/dropout_3/Mul_1Mul$lstm_4/lstm_cell_4/dropout_3/Mul:z:0%lstm_4/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_4/lstm_cell_4/dropout_3/Mul_1К
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dimƒ
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'lstm_4/lstm_cell_4/split/ReadVariableOpу
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_4/lstm_cell_4/splitґ
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMulЇ
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_1Ї
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_2Ї
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_3О
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_4/lstm_cell_4/split_1/split_dim∆
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_4/lstm_cell_4/split_1/ReadVariableOpл
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_4/lstm_cell_4/split_1њ
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd≈
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_1≈
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_2≈
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_3¶
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0$lstm_4/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mulђ
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_1ђ
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_2ђ
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_3≤
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02#
!lstm_4/lstm_cell_4/ReadVariableOp°
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_4/lstm_cell_4/strided_slice/stack•
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_4/lstm_cell_4/strided_slice/stack_1•
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_4/lstm_cell_4/strided_slice/stack_2о
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 lstm_4/lstm_cell_4/strided_sliceљ
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_4Ј
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/addС
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoidґ
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_1•
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_4/lstm_cell_4/strided_slice_1/stack©
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_4/lstm_cell_4/strided_slice_1/stack_1©
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_1/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_1Ѕ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_5љ
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_1Ч
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoid_1®
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_4ґ
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_2•
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_4/lstm_cell_4/strided_slice_2/stack©
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_4/lstm_cell_4/strided_slice_2/stack_1©
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_2/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_2Ѕ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_6љ
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_2К
lstm_4/lstm_cell_4/ReluRelulstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Reluі
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_5Ђ
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_3ґ
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_3•
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(lstm_4/lstm_cell_4/strided_slice_3/stack©
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_4/lstm_cell_4/strided_slice_3/stack_1©
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_3/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_3Ѕ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_7љ
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_4Ч
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoid_2О
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Relu_1Є
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_6Э
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2&
$lstm_4/TensorArrayV2_1/element_shape‘
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2_1\
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/timeН
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_4/while/maximum_iterationsx
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/while/loop_counterз
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_102642*$
condR
lstm_4_while_cond_102641*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_4/while√
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)lstm_4/TensorArrayV2Stack/TensorListStackП
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_4/strided_slice_3/stackК
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_4/strided_slice_3/stack_1К
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_3/stack_2ƒ
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_4/strided_slice_3З
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose_1/permЅ
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtime•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_6/MatMul/ReadVariableOp§
dense_6/MatMulMatMullstm_4/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddj
reshape_3/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_3/ShapeИ
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stackМ
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1М
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2Ю
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2“
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape£
reshape_3/ReshapeReshapedense_7/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_3/Reshapeм
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muly
IdentityIdentityreshape_3/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЙ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
£R
ж
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_100717

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
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

Identity_2Ж
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
а	
¶
-__inference_sequential_2_layer_call_fn_102867

inputs
unknown:	А
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_1021102
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®

ѕ
lstm_4_while_cond_102641*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_102641___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_102641___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_102641___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_102641___redundant_placeholder3
lstm_4_while_identity
У
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
lstm_4/while/Lessr
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_4/while/Identity"7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
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
’!
н
H__inference_sequential_2_layer_call_and_return_conditional_losses_102202
input_3 
lstm_4_102177:	А
lstm_4_102179:	А 
lstm_4_102181:	 А 
dense_6_102184:  
dense_6_102186:  
dense_7_102189: 
dense_7_102191:
identityИҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpЬ
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinput_3lstm_4_102177lstm_4_102179lstm_4_102181*
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1020522 
lstm_4/StatefulPartitionedCall∞
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_6_102184dense_6_102186*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1016452!
dense_6/StatefulPartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102189dense_7_102191*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1016612!
dense_7/StatefulPartitionedCallэ
reshape_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_1016802
reshape_3/PartitionedCall…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_102177*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_3/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityс
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
’
√
while_cond_103257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103257___redundant_placeholder04
0while_while_cond_103257___redundant_placeholder14
0while_while_cond_103257___redundant_placeholder24
0while_while_cond_103257___redundant_placeholder3
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
ђ
∆
__inference_loss_fn_0_104319W
Dlstm_4_lstm_cell_4_kernel_regularizer_square_readvariableop_resource:	А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpА
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDlstm_4_lstm_cell_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulw
IdentityIdentity-lstm_4/lstm_cell_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityМ
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp
њ%
№
while_body_100731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_100755_0:	А)
while_lstm_cell_4_100757_0:	А-
while_lstm_cell_4_100759_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_100755:	А'
while_lstm_cell_4_100757:	А+
while_lstm_cell_4_100759:	 АИҐ)while/lstm_cell_4/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_100755_0while_lstm_cell_4_100757_0while_lstm_cell_4_100759_0*
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1007172+
)while/lstm_cell_4/StatefulPartitionedCallц
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/Identity_3£
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
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
while_lstm_cell_4_100755while_lstm_cell_4_100755_0"6
while_lstm_cell_4_100757while_lstm_cell_4_100757_0"6
while_lstm_cell_4_100759while_lstm_cell_4_100759_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
і
х
,__inference_lstm_cell_4_layer_call_fn_104291

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall¬
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1007172
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
ї∞
Ш	
while_body_101887
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeЗ
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell_4/dropout/Const«
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeЙ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2≤Кё28
6while/lstm_cell_4/dropout/random_uniform/RandomUniformЩ
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell_4/dropout/GreaterEqual/yЖ
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout/Mul_1Л
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_1/ConstЌ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ўџЛ2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_1/GreaterEqual/yО
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_1/Mul_1Л
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_2/ConstЌ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeО
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2И©[2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_2/GreaterEqual/yО
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_2/Mul_1Л
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_3/ConstЌ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeП
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2УЋЦ2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_3/GreaterEqual/yО
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
ƒ~
Ш	
while_body_103533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
тћ
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_103973

inputs<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout/Constѓ
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeч
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2фЄк22
0lstm_cell_4/dropout/random_uniform/RandomUniformН
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell_4/dropout/GreaterEqual/yо
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_1/Constµ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeь
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2тЈ24
2lstm_cell_4/dropout_1/random_uniform/RandomUniformС
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_1/GreaterEqual/yц
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_2/Constµ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeэ
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2 €ƒ24
2lstm_cell_4/dropout_2/random_uniform/RandomUniformС
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_2/GreaterEqual/yц
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_3/Constµ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2АНµ24
2lstm_cell_4/dropout_3/random_uniform/RandomUniformС
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_3/GreaterEqual/yц
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_103808*
condR
while_cond_103807*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–Q
Њ
B__inference_lstm_4_layer_call_and_return_conditional_losses_101103

inputs%
lstm_cell_4_101015:	А!
lstm_cell_4_101017:	А%
lstm_cell_4_101019:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ#lstm_cell_4/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_101015lstm_cell_4_101017lstm_cell_4_101019*
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1009502%
#lstm_cell_4/StatefulPartitionedCallП
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
while/loop_counterљ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_101015lstm_cell_4_101017lstm_cell_4_101019*
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
while_body_101028*
condR
while_cond_101027*K
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
runtimeќ
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_4_101015*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

IdentityЇ
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–Q
Њ
B__inference_lstm_4_layer_call_and_return_conditional_losses_100806

inputs%
lstm_cell_4_100718:	А!
lstm_cell_4_100720:	А%
lstm_cell_4_100722:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ#lstm_cell_4/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_100718lstm_cell_4_100720lstm_cell_4_100722*
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1007172%
#lstm_cell_4/StatefulPartitionedCallП
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
while/loop_counterљ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_100718lstm_cell_4_100720lstm_cell_4_100722*
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
while_body_100731*
condR
while_cond_100730*K
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
runtimeќ
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_4_100718*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

IdentityЇ
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≥R
и
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104161

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
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

Identity_2Ж
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
∆
F
*__inference_reshape_3_layer_call_fn_104074

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
E__inference_reshape_3_layer_call_and_return_conditional_losses_1016802
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
ї∞
Ш	
while_body_103808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeЗ
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2!
while/lstm_cell_4/dropout/Const«
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeИ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ъ£-28
6while/lstm_cell_4/dropout/random_uniform/RandomUniformЩ
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2*
(while/lstm_cell_4/dropout/GreaterEqual/yЖ
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout/Mul_1Л
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_1/ConstЌ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Ёр≠2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_1/GreaterEqual/yО
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_1/Mul_1Л
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_2/ConstЌ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeП
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ѕќУ2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_2/GreaterEqual/yО
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_2/Mul_1Л
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2#
!while/lstm_cell_4/dropout_3/ConstЌ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeП
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2£Об2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformЭ
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2,
*while/lstm_cell_4/dropout_3/GreaterEqual/yО
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
while_cond_101886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101886___redundant_placeholder04
0while_while_cond_101886___redundant_placeholder14
0while_while_cond_101886___redundant_placeholder24
0while_while_cond_101886___redundant_placeholder3
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
і
х
,__inference_lstm_cell_4_layer_call_fn_104308

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identity

identity_1

identity_2ИҐStatefulPartitionedCall¬
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
GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1009502
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
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
Уv
и
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104274

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpZ
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
seed2ћТЊ2&
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
seed2вК!2(
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
seed2ОГЙ2(
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
seed2–юЯ2(
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
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

Identity_2Ж
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
while_cond_101492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101492___redundant_placeholder04
0while_while_cond_101492___redundant_placeholder14
0while_while_cond_101492___redundant_placeholder24
0while_while_cond_101492___redundant_placeholder3
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
Љ
ґ
'__inference_lstm_4_layer_call_fn_103995
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCallБ
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1011032
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
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ў—
Ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102500

inputsC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	АA
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	А=
*lstm_4_lstm_cell_4_readvariableop_resource:	 А8
&dense_6_matmul_readvariableop_resource:  5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identityИҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ!lstm_4/lstm_cell_4/ReadVariableOpҐ#lstm_4/lstm_cell_4/ReadVariableOp_1Ґ#lstm_4/lstm_cell_4/ReadVariableOp_2Ґ#lstm_4/lstm_cell_4/ReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ'lstm_4/lstm_cell_4/split/ReadVariableOpҐ)lstm_4/lstm_cell_4/split_1/ReadVariableOpҐlstm_4/whileR
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_4/ShapeВ
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stackЖ
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1Ж
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2М
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicej
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros/mul/yИ
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_4/zeros/Less/yГ
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessp
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros/packed/1Я
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/ConstС
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros_1/mul/yО
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_4/zeros_1/Less/yЛ
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lesst
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/zeros_1/packed/1•
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/ConstЩ
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/zeros_1Г
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/permП
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
lstm_4/transposed
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:2
lstm_4/Shape_1Ж
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_1/stackК
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_1К
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_2Ш
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slice_1У
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_4/TensorArrayV2/element_shapeќ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2Ќ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2>
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_4/TensorArrayUnstack/TensorListFromTensorЖ
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_2/stackК
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_1К
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_2¶
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
lstm_4/strided_slice_2Н
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/ones_like/ShapeН
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"lstm_4/lstm_cell_4/ones_like/Const–
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/ones_likeК
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dimƒ
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'lstm_4/lstm_cell_4/split/ReadVariableOpу
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_4/lstm_cell_4/splitґ
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMulЇ
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_1Ї
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_2Ї
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_3О
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_4/lstm_cell_4/split_1/split_dim∆
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_4/lstm_cell_4/split_1/ReadVariableOpл
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_4/lstm_cell_4/split_1њ
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd≈
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_1≈
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_2≈
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/BiasAdd_3І
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mulЂ
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_1Ђ
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_2Ђ
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_3≤
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02#
!lstm_4/lstm_cell_4/ReadVariableOp°
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_4/lstm_cell_4/strided_slice/stack•
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_4/lstm_cell_4/strided_slice/stack_1•
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_4/lstm_cell_4/strided_slice/stack_2о
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 lstm_4/lstm_cell_4/strided_sliceљ
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_4Ј
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/addС
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoidґ
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_1•
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_4/lstm_cell_4/strided_slice_1/stack©
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*lstm_4/lstm_cell_4/strided_slice_1/stack_1©
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_1/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_1Ѕ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_5љ
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_1Ч
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoid_1®
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_4ґ
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_2•
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_4/lstm_cell_4/strided_slice_2/stack©
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*lstm_4/lstm_cell_4/strided_slice_2/stack_1©
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_2/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_2Ѕ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_6љ
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_2К
lstm_4/lstm_cell_4/ReluRelulstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Reluі
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_5Ђ
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_3ґ
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_3•
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(lstm_4/lstm_cell_4/strided_slice_3/stack©
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_4/lstm_cell_4/strided_slice_3/stack_1©
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_4/lstm_cell_4/strided_slice_3/stack_2ъ
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_3Ѕ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/MatMul_7љ
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/add_4Ч
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Sigmoid_2О
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/Relu_1Є
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_4/lstm_cell_4/mul_6Э
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2&
$lstm_4/TensorArrayV2_1/element_shape‘
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2_1\
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/timeН
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_4/while/maximum_iterationsx
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/while/loop_counterз
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_102345*$
condR
lstm_4_while_cond_102344*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_4/while√
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)lstm_4/TensorArrayV2Stack/TensorListStackП
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_4/strided_slice_3/stackК
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_4/strided_slice_3/stack_1К
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_3/stack_2ƒ
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_4/strided_slice_3З
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose_1/permЅ
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtime•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_6/MatMul/ReadVariableOp§
dense_6/MatMulMatMullstm_4/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddj
reshape_3/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_3/ShapeИ
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stackМ
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1М
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2Ю
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2“
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape£
reshape_3/ReshapeReshapedense_7/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_3/Reshapeм
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muly
IdentityIdentityreshape_3/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityЙ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒ~
Ш	
while_body_101493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	 А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	 АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЙ
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/ShapeЛ
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!while/lstm_cell_4/ones_like/Constћ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02"
 while/lstm_cell_4/ReadVariableOpЯ
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stack£
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice/stack_1£
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2и
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2+
)while/lstm_cell_4/strided_slice_1/stack_1І
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2ф
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2+
)while/lstm_cell_4/strided_slice_2/stack_1І
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2ф
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2)
'while/lstm_cell_4/strided_slice_3/stackІ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1І
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2ф
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_4/mul_6я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
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
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5ј

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
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
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 
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
„Я
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_103666

inputs<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_103533*
condR
while_cond_103532*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
”
%sequential_2_lstm_4_while_cond_100443D
@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counterJ
Fsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations)
%sequential_2_lstm_4_while_placeholder+
'sequential_2_lstm_4_while_placeholder_1+
'sequential_2_lstm_4_while_placeholder_2+
'sequential_2_lstm_4_while_placeholder_3F
Bsequential_2_lstm_4_while_less_sequential_2_lstm_4_strided_slice_1\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_100443___redundant_placeholder0\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_100443___redundant_placeholder1\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_100443___redundant_placeholder2\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_100443___redundant_placeholder3&
"sequential_2_lstm_4_while_identity
‘
sequential_2/lstm_4/while/LessLess%sequential_2_lstm_4_while_placeholderBsequential_2_lstm_4_while_less_sequential_2_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_2/lstm_4/while/LessЩ
"sequential_2/lstm_4/while/IdentityIdentity"sequential_2/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_2/lstm_4/while/Identity"Q
"sequential_2_lstm_4_while_identity+sequential_2/lstm_4/while/Identity:output:0*(
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
„Я
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_101626

inputs<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_101493*
condR
while_cond_101492*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Дv
ж
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_100950

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	 А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpX
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
seed2®’П2&
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
seed2»“ћ2(
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
seed2¶Ы¬2(
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
seed2€…∆2(
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
:	А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
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
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
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

Identity_2Ж
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€ :€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
§
і
'__inference_lstm_4_layer_call_fn_104006

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	 А
identityИҐStatefulPartitionedCall€
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
GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1016262
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
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
г	
І
-__inference_sequential_2_layer_call_fn_101706
input_3
unknown:	А
	unknown_0:	А
	unknown_1:	 А
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_1016892
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
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_3
•

ф
C__inference_dense_7_layer_call_and_return_conditional_losses_101661

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
’
√
while_cond_102982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_102982___redundant_placeholder04
0while_while_cond_102982___redundant_placeholder14
0while_while_cond_102982___redundant_placeholder24
0while_while_cond_102982___redundant_placeholder3
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
тћ
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_102052

inputs<
)lstm_cell_4_split_readvariableop_resource:	А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐlstm_cell_4/ReadVariableOpҐlstm_cell_4/ReadVariableOp_1Ґlstm_cell_4/ReadVariableOp_2Ґlstm_cell_4/ReadVariableOp_3Ґ lstm_cell_4/split/ReadVariableOpҐ"lstm_cell_4/split_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€2
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
valueB"€€€€   27
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
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2x
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
lstm_cell_4/ones_like/Constі
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout/Constѓ
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeч
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2Ч†ї22
0lstm_cell_4/dropout/random_uniform/RandomUniformН
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2$
"lstm_cell_4/dropout/GreaterEqual/yо
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_1/Constµ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeэ
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ыѕђ24
2lstm_cell_4/dropout_1/random_uniform/RandomUniformС
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_1/GreaterEqual/yц
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_2/Constµ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeь
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2СЗ,24
2lstm_cell_4/dropout_2/random_uniform/RandomUniformС
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_2/GreaterEqual/yц
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
lstm_cell_4/dropout_3/Constµ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*
seed±€е)*
seed2ГљЋ24
2lstm_cell_4/dropout_3/random_uniform/RandomUniformС
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2&
$lstm_cell_4/dropout_3/GreaterEqual/yц
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOpУ
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stackЧ
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice/stack_1Ч
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2ƒ
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2%
#lstm_cell_4/strided_slice_1/stack_1Ы
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2–
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2%
#lstm_cell_4/strided_slice_2/stack_1Ы
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2–
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	 А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2#
!lstm_cell_4/strided_slice_3/stackЫ
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1Ы
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2–
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_4/mul_6П
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
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
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
while_body_101887*
condR
while_cond_101886*K
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
runtimeе
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2.
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareЂ
+lstm_4/lstm_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm_4/lstm_cell_4/kernel/Regularizer/Constж
)lstm_4/lstm_cell_4/kernel/Regularizer/SumSum0lstm_4/lstm_cell_4/kernel/Regularizer/Square:y:04lstm_4/lstm_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/SumЯ
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
?
input_34
serving_default_input_3:0€€€€€€€€€A
	reshape_34
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ї
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
*`&call_and_return_all_conditional_losses
a__call__
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
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_rnn_layer
ї

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
ї

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
•
trainable_variables
regularization_losses
	variables
 	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
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
 

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
є

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
 :  2dense_6/kernel
: 2dense_6/bias
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
 : 2dense_7/kernel
:2dense_7/bias
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
≠

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
≠

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
,:*	А2lstm_4/lstm_cell_4/kernel
6:4	 А2#lstm_4/lstm_cell_4/recurrent_kernel
&:$А2lstm_4/lstm_cell_4/bias
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
≠

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
%:#  2Adam/dense_6/kernel/m
: 2Adam/dense_6/bias/m
%:# 2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
1:/	А2 Adam/lstm_4/lstm_cell_4/kernel/m
;:9	 А2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
+:)А2Adam/lstm_4/lstm_cell_4/bias/m
%:#  2Adam/dense_6/kernel/v
: 2Adam/dense_6/bias/v
%:# 2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
1:/	А2 Adam/lstm_4/lstm_cell_4/kernel/v
;:9	 А2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
+:)А2Adam/lstm_4/lstm_cell_4/bias/v
о2л
H__inference_sequential_2_layer_call_and_return_conditional_losses_102500
H__inference_sequential_2_layer_call_and_return_conditional_losses_102829
H__inference_sequential_2_layer_call_and_return_conditional_losses_102174
H__inference_sequential_2_layer_call_and_return_conditional_losses_102202ј
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
В2€
-__inference_sequential_2_layer_call_fn_101706
-__inference_sequential_2_layer_call_fn_102848
-__inference_sequential_2_layer_call_fn_102867
-__inference_sequential_2_layer_call_fn_102146ј
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
!__inference__wrapped_model_100593input_3"Ш
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
л2и
B__inference_lstm_4_layer_call_and_return_conditional_losses_103116
B__inference_lstm_4_layer_call_and_return_conditional_losses_103423
B__inference_lstm_4_layer_call_and_return_conditional_losses_103666
B__inference_lstm_4_layer_call_and_return_conditional_losses_103973’
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
€2ь
'__inference_lstm_4_layer_call_fn_103984
'__inference_lstm_4_layer_call_fn_103995
'__inference_lstm_4_layer_call_fn_104006
'__inference_lstm_4_layer_call_fn_104017’
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
н2к
C__inference_dense_6_layer_call_and_return_conditional_losses_104028Ґ
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
“2ѕ
(__inference_dense_6_layer_call_fn_104037Ґ
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
н2к
C__inference_dense_7_layer_call_and_return_conditional_losses_104047Ґ
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
“2ѕ
(__inference_dense_7_layer_call_fn_104056Ґ
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_104069Ґ
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
*__inference_reshape_3_layer_call_fn_104074Ґ
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
ЋB»
$__inference_signature_wrapper_102235input_3"Ф
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
÷2”
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104161
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104274Њ
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
†2Э
,__inference_lstm_cell_4_layer_call_fn_104291
,__inference_lstm_cell_4_layer_call_fn_104308Њ
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
__inference_loss_fn_0_104319П
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
!__inference__wrapped_model_100593z&('4Ґ1
*Ґ'
%К"
input_3€€€€€€€€€
™ "9™6
4
	reshape_3'К$
	reshape_3€€€€€€€€€£
C__inference_dense_6_layer_call_and_return_conditional_losses_104028\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ {
(__inference_dense_6_layer_call_fn_104037O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ £
C__inference_dense_7_layer_call_and_return_conditional_losses_104047\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_7_layer_call_fn_104056O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€;
__inference_loss_fn_0_104319&Ґ

Ґ 
™ "К √
B__inference_lstm_4_layer_call_and_return_conditional_losses_103116}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ √
B__inference_lstm_4_layer_call_and_return_conditional_losses_103423}&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ≥
B__inference_lstm_4_layer_call_and_return_conditional_losses_103666m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ≥
B__inference_lstm_4_layer_call_and_return_conditional_losses_103973m&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ Ы
'__inference_lstm_4_layer_call_fn_103984p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ы
'__inference_lstm_4_layer_call_fn_103995p&('OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ Л
'__inference_lstm_4_layer_call_fn_104006`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Л
'__inference_lstm_4_layer_call_fn_104017`&('?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ …
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104161э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ …
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_104274э&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
Ъ Ю
,__inference_lstm_cell_4_layer_call_fn_104291н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
1/1€€€€€€€€€ Ю
,__inference_lstm_cell_4_layer_call_fn_104308н&('АҐ}
vҐs
 К
inputs€€€€€€€€€
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_104069\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ }
*__inference_reshape_3_layer_call_fn_104074O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Њ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102174r&('<Ґ9
2Ґ/
%К"
input_3€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Њ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102202r&('<Ґ9
2Ґ/
%К"
input_3€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102500q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_2_layer_call_and_return_conditional_losses_102829q&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ц
-__inference_sequential_2_layer_call_fn_101706e&('<Ґ9
2Ґ/
%К"
input_3€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ц
-__inference_sequential_2_layer_call_fn_102146e&('<Ґ9
2Ґ/
%К"
input_3€€€€€€€€€
p

 
™ "К€€€€€€€€€Х
-__inference_sequential_2_layer_call_fn_102848d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Х
-__inference_sequential_2_layer_call_fn_102867d&(';Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€Ѓ
$__inference_signature_wrapper_102235Е&('?Ґ<
Ґ 
5™2
0
input_3%К"
input_3€€€€€€€€€"9™6
4
	reshape_3'К$
	reshape_3€€€€€€€€€