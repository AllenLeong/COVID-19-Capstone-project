┴■'
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8▄Р&
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:А*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	А*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
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
lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А**
shared_namelstm_8/lstm_cell_8/kernel
И
-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/kernel*
_output_shapes
:	А*
dtype0
г
#lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*4
shared_name%#lstm_8/lstm_cell_8/recurrent_kernel
Ь
7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_8/lstm_cell_8/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_8/lstm_cell_8/bias
А
+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/bias*
_output_shapes	
:А*
dtype0
П
lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А**
shared_namelstm_9/lstm_cell_9/kernel
И
-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/kernel*
_output_shapes
:	@А*
dtype0
д
#lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*4
shared_name%#lstm_9/lstm_cell_9/recurrent_kernel
Э
7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_9/lstm_cell_9/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
З
lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_9/lstm_cell_9/bias
А
+lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/bias*
_output_shapes	
:А*
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
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_8/kernel/m
Б
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m* 
_output_shapes
:
АА*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_9/kernel/m
А
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	А*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
Э
 Adam/lstm_8/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/m
Ц
4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/m*
_output_shapes
:	А*
dtype0
▒
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
к
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_8/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_8/lstm_cell_8/bias/m
О
2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/m*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_9/lstm_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_9/lstm_cell_9/kernel/m
Ц
4Adam/lstm_9/lstm_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_9/lstm_cell_9/kernel/m*
_output_shapes
:	@А*
dtype0
▓
*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*;
shared_name,*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m
л
>Adam/lstm_9/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m* 
_output_shapes
:
АА*
dtype0
Х
Adam/lstm_9/lstm_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_9/lstm_cell_9/bias/m
О
2Adam/lstm_9/lstm_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_9/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_8/kernel/v
Б
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v* 
_output_shapes
:
АА*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_9/kernel/v
А
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	А*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
Э
 Adam/lstm_8/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/v
Ц
4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/v*
_output_shapes
:	А*
dtype0
▒
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
к
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_8/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_8/lstm_cell_8/bias/v
О
2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/v*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_9/lstm_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_9/lstm_cell_9/kernel/v
Ц
4Adam/lstm_9/lstm_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_9/lstm_cell_9/kernel/v*
_output_shapes
:	@А*
dtype0
▓
*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*;
shared_name,*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v
л
>Adam/lstm_9/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v* 
_output_shapes
:
АА*
dtype0
Х
Adam/lstm_9/lstm_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_9/lstm_cell_9/bias/v
О
2Adam/lstm_9/lstm_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_9/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Т?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*═>
value├>B└> B╣>
┤
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
R
.	variables
/regularization_losses
0trainable_variables
1	keras_api
И
2iter

3beta_1

4beta_2
	5decay
6learning_rate"mА#mБ(mВ)mГ7mД8mЕ9mЖ:mЗ;mИ<mЙ"vК#vЛ(vМ)vН7vО8vП9vР:vС;vТ<vУ
F
70
81
92
:3
;4
<5
"6
#7
(8
)9
 
F
70
81
92
:3
;4
<5
"6
#7
(8
)9
н
		variables

=layers
>non_trainable_variables
?metrics

regularization_losses
trainable_variables
@layer_regularization_losses
Alayer_metrics
 
О
B
state_size

7kernel
8recurrent_kernel
9bias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
 

70
81
92
 

70
81
92
╣
	variables

Glayers
Hnon_trainable_variables
Imetrics

Jstates
regularization_losses
trainable_variables
Klayer_regularization_losses
Llayer_metrics
 
 
 
н
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
regularization_losses
trainable_variables
Player_regularization_losses

Qlayers
О
R
state_size

:kernel
;recurrent_kernel
<bias
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
 

:0
;1
<2
 

:0
;1
<2
╣
	variables

Wlayers
Xnon_trainable_variables
Ymetrics

Zstates
regularization_losses
trainable_variables
[layer_regularization_losses
\layer_metrics
 
 
 
н
	variables
]layer_metrics
^non_trainable_variables
_metrics
regularization_losses
 trainable_variables
`layer_regularization_losses

alayers
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
н
$	variables
blayer_metrics
cnon_trainable_variables
dmetrics
%regularization_losses
&trainable_variables
elayer_regularization_losses

flayers
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
н
*	variables
glayer_metrics
hnon_trainable_variables
imetrics
+regularization_losses
,trainable_variables
jlayer_regularization_losses

klayers
 
 
 
н
.	variables
llayer_metrics
mnon_trainable_variables
nmetrics
/regularization_losses
0trainable_variables
olayer_regularization_losses

players
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
US
VARIABLE_VALUElstm_8/lstm_cell_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_8/lstm_cell_8/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_8/lstm_cell_8/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_9/lstm_cell_9/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_9/lstm_cell_9/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_9/lstm_cell_9/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 

q0
 
 
 

70
81
92
 

70
81
92
н
C	variables
rlayer_metrics
snon_trainable_variables
tmetrics
Dregularization_losses
Etrainable_variables
ulayer_regularization_losses

vlayers

0
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

:0
;1
<2
 

:0
;1
<2
н
S	variables
wlayer_metrics
xnon_trainable_variables
ymetrics
Tregularization_losses
Utrainable_variables
zlayer_regularization_losses

{layers

0
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
 
 
 
 
 
4
	|total
	}count
~	variables
	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

|0
}1

~	variables
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_9/lstm_cell_9/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_9/lstm_cell_9/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_9/lstm_cell_9/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_9/lstm_cell_9/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
З
serving_default_lstm_8_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
╜
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_8_inputlstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_102513
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
═
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_8/lstm_cell_8/bias/Read/ReadVariableOp-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp+lstm_9/lstm_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOp4Adam/lstm_9/lstm_cell_9/kernel/m/Read/ReadVariableOp>Adam/lstm_9/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_9/lstm_cell_9/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOp4Adam/lstm_9/lstm_cell_9/kernel/v/Read/ReadVariableOp>Adam/lstm_9/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_9/lstm_cell_9/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
__inference__traced_save_104958
ф	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biastotalcountAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m Adam/lstm_8/lstm_cell_8/kernel/m*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mAdam/lstm_8/lstm_cell_8/bias/m Adam/lstm_9/lstm_cell_9/kernel/m*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mAdam/lstm_9/lstm_cell_9/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v Adam/lstm_8/lstm_cell_8/kernel/v*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vAdam/lstm_8/lstm_cell_8/bias/v Adam/lstm_9/lstm_cell_9/kernel/v*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vAdam/lstm_9/lstm_cell_9/bias/v*1
Tin*
(2&*
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
"__inference__traced_restore_105079Щу$
╪>
╔
while_body_104007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╡[
Х
B__inference_lstm_9_layer_call_and_return_conditional_losses_104393

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
:         @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_104309*
condR
while_cond_104308*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╞
F
*__inference_reshape_4_layer_call_fn_104615

inputs
identity╟
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
GPU 2J 8В *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_1018682
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
·
Е
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104694

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
┤
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_104571

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
м

╧
lstm_9_while_cond_102777*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_102777___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_102777___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_102777___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_102777___redundant_placeholder3
lstm_9_while_identity
У
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
┤
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_101940

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┬>
╟
while_body_102219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
з
╡
'__inference_lstm_9_layer_call_fn_103929

inputs
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1018072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ы[
Ч
B__inference_lstm_9_layer_call_and_return_conditional_losses_104091
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
 :                  @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_104007*
condR
while_cond_104006*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
╜$
■
H__inference_sequential_2_layer_call_and_return_conditional_losses_102370

inputs 
lstm_8_102342:	А 
lstm_8_102344:	@А
lstm_8_102346:	А 
lstm_9_102350:	@А!
lstm_9_102352:
АА
lstm_9_102354:	А"
dense_8_102358:
АА
dense_8_102360:	А!
dense_9_102363:	А
dense_9_102365:
identityИвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallвlstm_8/StatefulPartitionedCallвlstm_9/StatefulPartitionedCallЯ
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_102342lstm_8_102344lstm_8_102346*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1023032 
lstm_8/StatefulPartitionedCallФ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1021362#
!dropout_4/StatefulPartitionedCall└
lstm_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0lstm_9_102350lstm_9_102352lstm_9_102354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1021072 
lstm_9/StatefulPartitionedCall╡
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1019402#
!dropout_5/StatefulPartitionedCall┤
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_102358dense_8_102360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1018332!
dense_8/StatefulPartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102363dense_9_102365*
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
GPU 2J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1018492!
dense_9/StatefulPartitionedCall¤
reshape_4/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_1018682
reshape_4/PartitionedCallБ
IdentityIdentity"reshape_4/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЬ
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┘
├
while_cond_102022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_102022___redundant_placeholder04
0while_while_cond_102022___redundant_placeholder14
0while_while_cond_102022___redundant_placeholder24
0while_while_cond_102022___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
┘
├
while_cond_104459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_104459___redundant_placeholder04
0while_while_cond_104459___redundant_placeholder14
0while_while_cond_104459___redundant_placeholder24
0while_while_cond_104459___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
╫
╢
'__inference_lstm_8_layer_call_fn_103243
inputs_0
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1005932
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
┘
├
while_cond_104006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_104006___redundant_placeholder04
0while_while_cond_104006___redundant_placeholder14
0while_while_cond_104006___redundant_placeholder24
0while_while_cond_104006___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
Д[
╞
$sequential_2_lstm_8_while_body_99970D
@sequential_2_lstm_8_while_sequential_2_lstm_8_while_loop_counterJ
Fsequential_2_lstm_8_while_sequential_2_lstm_8_while_maximum_iterations)
%sequential_2_lstm_8_while_placeholder+
'sequential_2_lstm_8_while_placeholder_1+
'sequential_2_lstm_8_while_placeholder_2+
'sequential_2_lstm_8_while_placeholder_3C
?sequential_2_lstm_8_while_sequential_2_lstm_8_strided_slice_1_0
{sequential_2_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_8_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_2_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	А[
Hsequential_2_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АV
Gsequential_2_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	А&
"sequential_2_lstm_8_while_identity(
$sequential_2_lstm_8_while_identity_1(
$sequential_2_lstm_8_while_identity_2(
$sequential_2_lstm_8_while_identity_3(
$sequential_2_lstm_8_while_identity_4(
$sequential_2_lstm_8_while_identity_5A
=sequential_2_lstm_8_while_sequential_2_lstm_8_strided_slice_1}
ysequential_2_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_8_tensorarrayunstack_tensorlistfromtensorW
Dsequential_2_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	АY
Fsequential_2_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	@АT
Esequential_2_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	АИв<sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpв;sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpв=sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpы
Ksequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2M
Ksequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape╦
=sequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_2_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_8_tensorarrayunstack_tensorlistfromtensor_0%sequential_2_lstm_8_while_placeholderTsequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02?
=sequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItemВ
;sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpFsequential_2_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02=
;sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpд
,sequential_2/lstm_8/while/lstm_cell_8/MatMulMatMulDsequential_2/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,sequential_2/lstm_8/while/lstm_cell_8/MatMulИ
=sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpHsequential_2_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02?
=sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpН
.sequential_2/lstm_8/while/lstm_cell_8/MatMul_1MatMul'sequential_2_lstm_8_while_placeholder_2Esequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А20
.sequential_2/lstm_8/while/lstm_cell_8/MatMul_1Д
)sequential_2/lstm_8/while/lstm_cell_8/addAddV26sequential_2/lstm_8/while/lstm_cell_8/MatMul:product:08sequential_2/lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2+
)sequential_2/lstm_8/while/lstm_cell_8/addБ
<sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpGsequential_2_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpС
-sequential_2/lstm_8/while/lstm_cell_8/BiasAddBiasAdd-sequential_2/lstm_8/while/lstm_cell_8/add:z:0Dsequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-sequential_2/lstm_8/while/lstm_cell_8/BiasAdd░
5sequential_2/lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_2/lstm_8/while/lstm_cell_8/split/split_dim╫
+sequential_2/lstm_8/while/lstm_cell_8/splitSplit>sequential_2/lstm_8/while/lstm_cell_8/split/split_dim:output:06sequential_2/lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2-
+sequential_2/lstm_8/while/lstm_cell_8/split╤
-sequential_2/lstm_8/while/lstm_cell_8/SigmoidSigmoid4sequential_2/lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2/
-sequential_2/lstm_8/while/lstm_cell_8/Sigmoid╒
/sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid4sequential_2/lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @21
/sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_1э
)sequential_2/lstm_8/while/lstm_cell_8/mulMul3sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_1:y:0'sequential_2_lstm_8_while_placeholder_3*
T0*'
_output_shapes
:         @2+
)sequential_2/lstm_8/while/lstm_cell_8/mul╚
*sequential_2/lstm_8/while/lstm_cell_8/ReluRelu4sequential_2/lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2,
*sequential_2/lstm_8/while/lstm_cell_8/ReluА
+sequential_2/lstm_8/while/lstm_cell_8/mul_1Mul1sequential_2/lstm_8/while/lstm_cell_8/Sigmoid:y:08sequential_2/lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2-
+sequential_2/lstm_8/while/lstm_cell_8/mul_1ї
+sequential_2/lstm_8/while/lstm_cell_8/add_1AddV2-sequential_2/lstm_8/while/lstm_cell_8/mul:z:0/sequential_2/lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2-
+sequential_2/lstm_8/while/lstm_cell_8/add_1╒
/sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid4sequential_2/lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @21
/sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_2╟
,sequential_2/lstm_8/while/lstm_cell_8/Relu_1Relu/sequential_2/lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2.
,sequential_2/lstm_8/while/lstm_cell_8/Relu_1Д
+sequential_2/lstm_8/while/lstm_cell_8/mul_2Mul3sequential_2/lstm_8/while/lstm_cell_8/Sigmoid_2:y:0:sequential_2/lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2-
+sequential_2/lstm_8/while/lstm_cell_8/mul_2├
>sequential_2/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_2_lstm_8_while_placeholder_1%sequential_2_lstm_8_while_placeholder/sequential_2/lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_2/lstm_8/while/TensorArrayV2Write/TensorListSetItemД
sequential_2/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_2/lstm_8/while/add/y╣
sequential_2/lstm_8/while/addAddV2%sequential_2_lstm_8_while_placeholder(sequential_2/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_8/while/addИ
!sequential_2/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_2/lstm_8/while/add_1/y┌
sequential_2/lstm_8/while/add_1AddV2@sequential_2_lstm_8_while_sequential_2_lstm_8_while_loop_counter*sequential_2/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_8/while/add_1╗
"sequential_2/lstm_8/while/IdentityIdentity#sequential_2/lstm_8/while/add_1:z:0^sequential_2/lstm_8/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_2/lstm_8/while/Identityт
$sequential_2/lstm_8/while/Identity_1IdentityFsequential_2_lstm_8_while_sequential_2_lstm_8_while_maximum_iterations^sequential_2/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_8/while/Identity_1╜
$sequential_2/lstm_8/while/Identity_2Identity!sequential_2/lstm_8/while/add:z:0^sequential_2/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_8/while/Identity_2ъ
$sequential_2/lstm_8/while/Identity_3IdentityNsequential_2/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_8/while/Identity_3▄
$sequential_2/lstm_8/while/Identity_4Identity/sequential_2/lstm_8/while/lstm_cell_8/mul_2:z:0^sequential_2/lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2&
$sequential_2/lstm_8/while/Identity_4▄
$sequential_2/lstm_8/while/Identity_5Identity/sequential_2/lstm_8/while/lstm_cell_8/add_1:z:0^sequential_2/lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2&
$sequential_2/lstm_8/while/Identity_5┐
sequential_2/lstm_8/while/NoOpNoOp=^sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<^sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp>^sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_2/lstm_8/while/NoOp"Q
"sequential_2_lstm_8_while_identity+sequential_2/lstm_8/while/Identity:output:0"U
$sequential_2_lstm_8_while_identity_1-sequential_2/lstm_8/while/Identity_1:output:0"U
$sequential_2_lstm_8_while_identity_2-sequential_2/lstm_8/while/Identity_2:output:0"U
$sequential_2_lstm_8_while_identity_3-sequential_2/lstm_8/while/Identity_3:output:0"U
$sequential_2_lstm_8_while_identity_4-sequential_2/lstm_8/while/Identity_4:output:0"U
$sequential_2_lstm_8_while_identity_5-sequential_2/lstm_8/while/Identity_5:output:0"Р
Esequential_2_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resourceGsequential_2_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"Т
Fsequential_2_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resourceHsequential_2_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"О
Dsequential_2_lstm_8_while_lstm_cell_8_matmul_readvariableop_resourceFsequential_2_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"А
=sequential_2_lstm_8_while_sequential_2_lstm_8_strided_slice_1?sequential_2_lstm_8_while_sequential_2_lstm_8_strided_slice_1_0"°
ysequential_2_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_8_tensorarrayunstack_tensorlistfromtensor{sequential_2_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2|
<sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<sequential_2/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2z
;sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp;sequential_2/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2~
=sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp=sequential_2/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
┘
├
while_cond_104157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_104157___redundant_placeholder04
0while_while_cond_104157___redundant_placeholder14
0while_while_cond_104157___redundant_placeholder24
0while_while_cond_104157___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
У[
Ф
B__inference_lstm_8_layer_call_and_return_conditional_losses_103869

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
:         2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_103785*
condR
while_cond_103784*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
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
:         @*
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
:         @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ЬF
Б
B__inference_lstm_9_layer_call_and_return_conditional_losses_101223

inputs%
lstm_cell_9_101141:	@А&
lstm_cell_9_101143:
АА!
lstm_cell_9_101145:	А
identityИв#lstm_cell_9/StatefulPartitionedCallвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
 :                  @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2Ъ
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_101141lstm_cell_9_101143lstm_cell_9_101145*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1010762%
#lstm_cell_9/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counter┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_101141lstm_cell_9_101143lstm_cell_9_101145*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_101154*
condR
while_cond_101153*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity|
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
И
Д
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_101076

inputs

states
states_11
matmul_readvariableop_resource:	@А4
 matmul_1_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         А2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         А2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         А2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         А2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         А2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         А2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         А2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         А2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_namestates:PL
(
_output_shapes
:         А
 
_user_specified_namestates
И
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_101868

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
╒
├
while_cond_103331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103331___redundant_placeholder04
0while_while_cond_103331___redundant_placeholder14
0while_while_cond_103331___redundant_placeholder24
0while_while_cond_103331___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
┐
╖
'__inference_lstm_9_layer_call_fn_103918
inputs_0
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1012232
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
┐
╖
'__inference_lstm_9_layer_call_fn_103907
inputs_0
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1010132
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
╡[
Х
B__inference_lstm_9_layer_call_and_return_conditional_losses_104544

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
:         @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_104460*
condR
while_cond_104459*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Є
Ц
(__inference_dense_9_layer_call_fn_104600

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCallє
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
GPU 2J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1018492
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
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
и

╧
lstm_8_while_cond_102629*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_102629___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_102629___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_102629___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_102629___redundant_placeholder3
lstm_8_while_identity
У
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
И
Д
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_100930

inputs

states
states_11
matmul_readvariableop_resource:	@А4
 matmul_1_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         А2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         А2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         А2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         А2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         А2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         А2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         А2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         А2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_namestates:PL
(
_output_shapes
:         А
 
_user_specified_namestates
╣!
╢
H__inference_sequential_2_layer_call_and_return_conditional_losses_101871

inputs 
lstm_8_101643:	А 
lstm_8_101645:	@А
lstm_8_101647:	А 
lstm_9_101808:	@А!
lstm_9_101810:
АА
lstm_9_101812:	А"
dense_8_101834:
АА
dense_8_101836:	А!
dense_9_101850:	А
dense_9_101852:
identityИвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallвlstm_8/StatefulPartitionedCallвlstm_9/StatefulPartitionedCallЯ
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_101643lstm_8_101645lstm_8_101647*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1016422 
lstm_8/StatefulPartitionedCall№
dropout_4/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1016552
dropout_4/PartitionedCall╕
lstm_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0lstm_9_101808lstm_9_101810lstm_9_101812*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1018072 
lstm_9/StatefulPartitionedCall∙
dropout_5/PartitionedCallPartitionedCall'lstm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1018202
dropout_5/PartitionedCallм
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_101834dense_8_101836*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1018332!
dense_8/StatefulPartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_101850dense_9_101852*
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
GPU 2J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1018492!
dense_9/StatefulPartitionedCall¤
reshape_4/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_1018682
reshape_4/PartitionedCallБ
IdentityIdentity"reshape_4/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╘
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╒
├
while_cond_100313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100313___redundant_placeholder04
0while_while_cond_100313___redundant_placeholder14
0while_while_cond_100313___redundant_placeholder24
0while_while_cond_100313___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
┬>
╟
while_body_103483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╬
F
*__inference_dropout_4_layer_call_fn_103874

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1016552
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╠
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_102136

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Ў
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_104559

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┘
├
while_cond_100943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100943___redundant_placeholder04
0while_while_cond_100943___redundant_placeholder14
0while_while_cond_100943___redundant_placeholder24
0while_while_cond_100943___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
┘
├
while_cond_104308
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_104308___redundant_placeholder04
0while_while_cond_104308___redundant_placeholder14
0while_while_cond_104308___redundant_placeholder24
0while_while_cond_104308___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
╪>
╔
while_body_104309
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╩%
▐
while_body_101154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_9_101178_0:	@А.
while_lstm_cell_9_101180_0:
АА)
while_lstm_cell_9_101182_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_9_101178:	@А,
while_lstm_cell_9_101180:
АА'
while_lstm_cell_9_101182:	АИв)while/lstm_cell_9/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem▐
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_101178_0while_lstm_cell_9_101180_0while_lstm_cell_9_101182_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1010762+
)while/lstm_cell_9/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4д
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
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
while_lstm_cell_9_101178while_lstm_cell_9_101178_0"6
while_lstm_cell_9_101180while_lstm_cell_9_101180_0"6
while_lstm_cell_9_101182while_lstm_cell_9_101182_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╫H
й

lstm_9_while_body_102778*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	@АO
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААI
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	@АM
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
ААG
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	АИв/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpв.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpв0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp╤
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItem█
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpЁ
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_9/while/lstm_cell_9/MatMulт
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype022
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp┘
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_9/while/lstm_cell_9/MatMul_1╨
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/add┌
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp▌
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_9/while/lstm_cell_9/BiasAddЦ
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_9/while/lstm_cell_9/split/split_dimз
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2 
lstm_9/while/lstm_cell_9/splitл
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2"
 lstm_9/while/lstm_cell_9/Sigmoidп
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2$
"lstm_9/while/lstm_cell_9/Sigmoid_1║
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/mulв
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/Relu═
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/mul_1┬
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/add_1п
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2$
"lstm_9/while/lstm_cell_9/Sigmoid_2б
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2!
lstm_9/while/lstm_cell_9/Relu_1╤
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/mul_2В
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/yЕ
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/yЩ
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1З
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identityб
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1Й
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2╢
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3й
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2
lstm_9/while/Identity_4й
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2
lstm_9/while/Identity_5■
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_9/while/NoOp"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"─
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
ш

Б
-__inference_sequential_2_layer_call_fn_102563

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1023702
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
│
╙
%sequential_2_lstm_9_while_cond_100117D
@sequential_2_lstm_9_while_sequential_2_lstm_9_while_loop_counterJ
Fsequential_2_lstm_9_while_sequential_2_lstm_9_while_maximum_iterations)
%sequential_2_lstm_9_while_placeholder+
'sequential_2_lstm_9_while_placeholder_1+
'sequential_2_lstm_9_while_placeholder_2+
'sequential_2_lstm_9_while_placeholder_3F
Bsequential_2_lstm_9_while_less_sequential_2_lstm_9_strided_slice_1\
Xsequential_2_lstm_9_while_sequential_2_lstm_9_while_cond_100117___redundant_placeholder0\
Xsequential_2_lstm_9_while_sequential_2_lstm_9_while_cond_100117___redundant_placeholder1\
Xsequential_2_lstm_9_while_sequential_2_lstm_9_while_cond_100117___redundant_placeholder2\
Xsequential_2_lstm_9_while_sequential_2_lstm_9_while_cond_100117___redundant_placeholder3&
"sequential_2_lstm_9_while_identity
╘
sequential_2/lstm_9/while/LessLess%sequential_2_lstm_9_while_placeholderBsequential_2_lstm_9_while_less_sequential_2_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_2/lstm_9/while/LessЩ
"sequential_2/lstm_9/while/IdentityIdentity"sequential_2/lstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_2/lstm_9/while/Identity"Q
"sequential_2_lstm_9_while_identity+sequential_2/lstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
╒
├
while_cond_102218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_102218___redundant_placeholder04
0while_while_cond_102218___redundant_placeholder14
0while_while_cond_102218___redundant_placeholder24
0while_while_cond_102218___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
й

ї
C__inference_dense_9_layer_call_and_return_conditional_losses_101849

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┘
├
while_cond_101153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101153___redundant_placeholder04
0while_while_cond_101153___redundant_placeholder14
0while_while_cond_101153___redundant_placeholder24
0while_while_cond_101153___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
Є
Г
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_100300

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates
и

╧
lstm_8_while_cond_102951*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_102951___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_102951___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_102951___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_102951___redundant_placeholder3
lstm_8_while_identity
У
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
У[
Ф
B__inference_lstm_8_layer_call_and_return_conditional_losses_102303

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
:         2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_102219*
condR
while_cond_102218*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
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
:         @*
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
:         @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╒
├
while_cond_100523
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100523___redundant_placeholder04
0while_while_cond_100523___redundant_placeholder14
0while_while_cond_100523___redundant_placeholder24
0while_while_cond_100523___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
┘
├
while_cond_101722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101722___redundant_placeholder04
0while_while_cond_101722___redundant_placeholder14
0while_while_cond_101722___redundant_placeholder24
0while_while_cond_101722___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
┬>
╟
while_body_103634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╦!
╝
H__inference_sequential_2_layer_call_and_return_conditional_losses_102449
lstm_8_input 
lstm_8_102421:	А 
lstm_8_102423:	@А
lstm_8_102425:	А 
lstm_9_102429:	@А!
lstm_9_102431:
АА
lstm_9_102433:	А"
dense_8_102437:
АА
dense_8_102439:	А!
dense_9_102442:	А
dense_9_102444:
identityИвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallвlstm_8/StatefulPartitionedCallвlstm_9/StatefulPartitionedCallе
lstm_8/StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputlstm_8_102421lstm_8_102423lstm_8_102425*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1016422 
lstm_8/StatefulPartitionedCall№
dropout_4/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1016552
dropout_4/PartitionedCall╕
lstm_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0lstm_9_102429lstm_9_102431lstm_9_102433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1018072 
lstm_9/StatefulPartitionedCall∙
dropout_5/PartitionedCallPartitionedCall'lstm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1018202
dropout_5/PartitionedCallм
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_102437dense_8_102439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1018332!
dense_8/StatefulPartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102442dense_9_102444*
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
GPU 2J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1018492!
dense_9/StatefulPartitionedCall¤
reshape_4/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_1018682
reshape_4/PartitionedCallБ
IdentityIdentity"reshape_4/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╘
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
░
c
*__inference_dropout_4_layer_call_fn_103879

inputs
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1021362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ы[
Ч
B__inference_lstm_9_layer_call_and_return_conditional_losses_104242
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
 :                  @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_104158*
condR
while_cond_104157*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
В
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_101655

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         @2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
·
Е
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104726

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
╒
├
while_cond_103633
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103633___redundant_placeholder04
0while_while_cond_103633___redundant_placeholder14
0while_while_cond_103633___redundant_placeholder24
0while_while_cond_103633___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
м
┤
'__inference_lstm_8_layer_call_fn_103254

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1016422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╩

■
$__inference_signature_wrapper_102513
lstm_8_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_1002252
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
╥[
Ц
B__inference_lstm_8_layer_call_and_return_conditional_losses_103567
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileF
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
 :                  2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_103483*
condR
while_cond_103482*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
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
:         @*
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
 :                  @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╥[
Ц
B__inference_lstm_8_layer_call_and_return_conditional_losses_103416
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileF
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
 :                  2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_103332*
condR
while_cond_103331*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
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
:         @*
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
 :                  @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
┐%
▄
while_body_100524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_100548_0:	А-
while_lstm_cell_8_100550_0:	@А)
while_lstm_cell_8_100552_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_100548:	А+
while_lstm_cell_8_100550:	@А'
while_lstm_cell_8_100552:	АИв)while/lstm_cell_8/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem█
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_100548_0while_lstm_cell_8_100550_0while_lstm_cell_8_100552_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1004462+
)while/lstm_cell_8/StatefulPartitionedCallЎ
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5Ж

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
while_lstm_cell_8_100548while_lstm_cell_8_100548_0"6
while_lstm_cell_8_100550while_lstm_cell_8_100550_0"6
while_lstm_cell_8_100552while_lstm_cell_8_100552_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2V
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
ОF
А
B__inference_lstm_8_layer_call_and_return_conditional_losses_100383

inputs%
lstm_cell_8_100301:	А%
lstm_cell_8_100303:	@А!
lstm_cell_8_100305:	А
identityИв#lstm_cell_8/StatefulPartitionedCallвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
 :                  2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_100301lstm_cell_8_100303lstm_cell_8_100305*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1003002%
#lstm_cell_8/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counter╜
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_100301lstm_cell_8_100303lstm_cell_8_100305*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100314*
condR
while_cond_100313*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
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
:         @*
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
 :                  @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ў
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_101820

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ў
Ш
(__inference_dense_8_layer_call_fn_104580

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1018332
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧$
Д
H__inference_sequential_2_layer_call_and_return_conditional_losses_102480
lstm_8_input 
lstm_8_102452:	А 
lstm_8_102454:	@А
lstm_8_102456:	А 
lstm_9_102460:	@А!
lstm_9_102462:
АА
lstm_9_102464:	А"
dense_8_102468:
АА
dense_8_102470:	А!
dense_9_102473:	А
dense_9_102475:
identityИвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallвlstm_8/StatefulPartitionedCallвlstm_9/StatefulPartitionedCallе
lstm_8/StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputlstm_8_102452lstm_8_102454lstm_8_102456*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1023032 
lstm_8/StatefulPartitionedCallФ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1021362#
!dropout_4/StatefulPartitionedCall└
lstm_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0lstm_9_102460lstm_9_102462lstm_9_102464*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1021072 
lstm_9/StatefulPartitionedCall╡
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1019402#
!dropout_5/StatefulPartitionedCall┤
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_102468dense_8_102470*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1018332!
dense_8/StatefulPartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102473dense_9_102475*
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
GPU 2J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1018492!
dense_9/StatefulPartitionedCall¤
reshape_4/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_1018682
reshape_4/PartitionedCallБ
IdentityIdentity"reshape_4/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЬ
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
┴H
з

lstm_8_while_body_102630*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	АN
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	АL
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	@АG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	АИв/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpв.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpв0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp╤
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem█
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpЁ
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_8/while/lstm_cell_8/MatMulс
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp┘
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_8/while/lstm_cell_8/MatMul_1╨
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_8/while/lstm_cell_8/add┌
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp▌
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_8/while/lstm_cell_8/BiasAddЦ
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dimг
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2 
lstm_8/while/lstm_cell_8/splitк
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2"
 lstm_8/while/lstm_cell_8/Sigmoidо
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2$
"lstm_8/while/lstm_cell_8/Sigmoid_1╣
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:         @2
lstm_8/while/lstm_cell_8/mulб
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_8/while/lstm_cell_8/Relu╠
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/mul_1┴
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/add_1о
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2$
"lstm_8/while/lstm_cell_8/Sigmoid_2а
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2!
lstm_8/while/lstm_cell_8/Relu_1╨
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/mul_2В
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/yЕ
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
lstm_8/while/add_1/yЩ
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1З
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identityб
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1Й
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2╢
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3и
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_8/while/Identity_4и
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_8/while/Identity_5■
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
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
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"─
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
┬>
╟
while_body_101558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
В
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_103884

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         @2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ОF
А
B__inference_lstm_8_layer_call_and_return_conditional_losses_100593

inputs%
lstm_cell_8_100511:	А%
lstm_cell_8_100513:	@А!
lstm_cell_8_100515:	А
identityИв#lstm_cell_8/StatefulPartitionedCallвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
 :                  2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_100511lstm_cell_8_100513lstm_cell_8_100515*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1004462%
#lstm_cell_8/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counter╜
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_100511lstm_cell_8_100513lstm_cell_8_100515*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100524*
condR
while_cond_100523*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
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
:         @*
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
 :                  @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
У[
Ф
B__inference_lstm_8_layer_call_and_return_conditional_losses_103718

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
:         2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_103634*
condR
while_cond_103633*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
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
:         @*
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
:         @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Нв
┐
"__inference__traced_restore_105079
file_prefix3
assignvariableop_dense_8_kernel:
АА.
assignvariableop_1_dense_8_bias:	А4
!assignvariableop_2_dense_9_kernel:	А-
assignvariableop_3_dense_9_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
,assignvariableop_9_lstm_8_lstm_cell_8_kernel:	АJ
7assignvariableop_10_lstm_8_lstm_cell_8_recurrent_kernel:	@А:
+assignvariableop_11_lstm_8_lstm_cell_8_bias:	А@
-assignvariableop_12_lstm_9_lstm_cell_9_kernel:	@АK
7assignvariableop_13_lstm_9_lstm_cell_9_recurrent_kernel:
АА:
+assignvariableop_14_lstm_9_lstm_cell_9_bias:	А#
assignvariableop_15_total: #
assignvariableop_16_count: =
)assignvariableop_17_adam_dense_8_kernel_m:
АА6
'assignvariableop_18_adam_dense_8_bias_m:	А<
)assignvariableop_19_adam_dense_9_kernel_m:	А5
'assignvariableop_20_adam_dense_9_bias_m:G
4assignvariableop_21_adam_lstm_8_lstm_cell_8_kernel_m:	АQ
>assignvariableop_22_adam_lstm_8_lstm_cell_8_recurrent_kernel_m:	@АA
2assignvariableop_23_adam_lstm_8_lstm_cell_8_bias_m:	АG
4assignvariableop_24_adam_lstm_9_lstm_cell_9_kernel_m:	@АR
>assignvariableop_25_adam_lstm_9_lstm_cell_9_recurrent_kernel_m:
ААA
2assignvariableop_26_adam_lstm_9_lstm_cell_9_bias_m:	А=
)assignvariableop_27_adam_dense_8_kernel_v:
АА6
'assignvariableop_28_adam_dense_8_bias_v:	А<
)assignvariableop_29_adam_dense_9_kernel_v:	А5
'assignvariableop_30_adam_dense_9_bias_v:G
4assignvariableop_31_adam_lstm_8_lstm_cell_8_kernel_v:	АQ
>assignvariableop_32_adam_lstm_8_lstm_cell_8_recurrent_kernel_v:	@АA
2assignvariableop_33_adam_lstm_8_lstm_cell_8_bias_v:	АG
4assignvariableop_34_adam_lstm_9_lstm_cell_9_kernel_v:	@АR
>assignvariableop_35_adam_lstm_9_lstm_cell_9_recurrent_kernel_v:
ААA
2assignvariableop_36_adam_lstm_9_lstm_cell_9_bias_v:	А
identity_38ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9М
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Ш
valueОBЛ&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names┌
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ж
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9▒
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_8_lstm_cell_8_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┐
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_8_lstm_cell_8_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11│
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_8_lstm_cell_8_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╡
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_9_lstm_cell_9_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13┐
AssignVariableOp_13AssignVariableOp7assignvariableop_13_lstm_9_lstm_cell_9_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14│
AssignVariableOp_14AssignVariableOp+assignvariableop_14_lstm_9_lstm_cell_9_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15б
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16б
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▒
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18п
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_8_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19▒
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_9_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20п
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_9_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╝
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_8_lstm_cell_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╞
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_lstm_8_lstm_cell_8_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23║
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_lstm_8_lstm_cell_8_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_9_lstm_cell_9_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╞
AssignVariableOp_25AssignVariableOp>assignvariableop_25_adam_lstm_9_lstm_cell_9_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26║
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_lstm_9_lstm_cell_9_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▒
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_8_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28п
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_8_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▒
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_9_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30п
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_9_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╝
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_8_lstm_cell_8_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╞
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_lstm_8_lstm_cell_8_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33║
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_lstm_8_lstm_cell_8_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╝
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_9_lstm_cell_9_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╞
AssignVariableOp_35AssignVariableOp>assignvariableop_35_adam_lstm_9_lstm_cell_9_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36║
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_lstm_9_lstm_cell_9_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpМ
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37f
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_38Ї
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
к
╬
$sequential_2_lstm_8_while_cond_99969D
@sequential_2_lstm_8_while_sequential_2_lstm_8_while_loop_counterJ
Fsequential_2_lstm_8_while_sequential_2_lstm_8_while_maximum_iterations)
%sequential_2_lstm_8_while_placeholder+
'sequential_2_lstm_8_while_placeholder_1+
'sequential_2_lstm_8_while_placeholder_2+
'sequential_2_lstm_8_while_placeholder_3F
Bsequential_2_lstm_8_while_less_sequential_2_lstm_8_strided_slice_1[
Wsequential_2_lstm_8_while_sequential_2_lstm_8_while_cond_99969___redundant_placeholder0[
Wsequential_2_lstm_8_while_sequential_2_lstm_8_while_cond_99969___redundant_placeholder1[
Wsequential_2_lstm_8_while_sequential_2_lstm_8_while_cond_99969___redundant_placeholder2[
Wsequential_2_lstm_8_while_sequential_2_lstm_8_while_cond_99969___redundant_placeholder3&
"sequential_2_lstm_8_while_identity
╘
sequential_2/lstm_8/while/LessLess%sequential_2_lstm_8_while_placeholderBsequential_2_lstm_8_while_less_sequential_2_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_2/lstm_8/while/LessЩ
"sequential_2/lstm_8/while/IdentityIdentity"sequential_2/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_2/lstm_8/while/Identity"Q
"sequential_2_lstm_8_while_identity+sequential_2/lstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╫
╢
'__inference_lstm_8_layer_call_fn_103232
inputs_0
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1003832
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╩%
▐
while_body_100944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_9_100968_0:	@А.
while_lstm_cell_9_100970_0:
АА)
while_lstm_cell_9_100972_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_9_100968:	@А,
while_lstm_cell_9_100970:
АА'
while_lstm_cell_9_100972:	АИв)while/lstm_cell_9/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem▐
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_100968_0while_lstm_cell_9_100970_0while_lstm_cell_9_100972_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1009302+
)while/lstm_cell_9/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4д
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
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
while_lstm_cell_9_100968while_lstm_cell_9_100968_0"6
while_lstm_cell_9_100970while_lstm_cell_9_100970_0"6
while_lstm_cell_9_100972while_lstm_cell_9_100972_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╪>
╔
while_body_104158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
Н
ў
C__inference_dense_8_layer_call_and_return_conditional_losses_104591

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
й

ї
C__inference_dense_9_layer_call_and_return_conditional_losses_104610

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫H
й

lstm_9_while_body_103107*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	@АO
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААI
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	@АM
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
ААG
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	АИв/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpв.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpв0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp╤
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItem█
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpЁ
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_9/while/lstm_cell_9/MatMulт
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype022
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp┘
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_9/while/lstm_cell_9/MatMul_1╨
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/add┌
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp▌
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_9/while/lstm_cell_9/BiasAddЦ
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_9/while/lstm_cell_9/split/split_dimз
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2 
lstm_9/while/lstm_cell_9/splitл
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2"
 lstm_9/while/lstm_cell_9/Sigmoidп
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2$
"lstm_9/while/lstm_cell_9/Sigmoid_1║
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/mulв
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_9/while/lstm_cell_9/Relu═
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/mul_1┬
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/add_1п
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2$
"lstm_9/while/lstm_cell_9/Sigmoid_2б
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2!
lstm_9/while/lstm_cell_9/Relu_1╤
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2 
lstm_9/while/lstm_cell_9/mul_2В
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/yЕ
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/yЩ
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1З
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identityб
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1Й
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2╢
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3й
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2
lstm_9/while/Identity_4й
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2
lstm_9/while/Identity_5■
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_9/while/NoOp"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"─
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╪>
╔
while_body_101723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
ш

Б
-__inference_sequential_2_layer_call_fn_102538

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1018712
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╡[
Х
B__inference_lstm_9_layer_call_and_return_conditional_losses_101807

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
:         @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_101723*
condR
while_cond_101722*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╠
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_103896

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
з
╡
'__inference_lstm_9_layer_call_fn_103940

inputs
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_1021072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
У[
Ф
B__inference_lstm_8_layer_call_and_return_conditional_losses_101642

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	А?
,lstm_cell_8_matmul_1_readvariableop_resource:	@А:
+lstm_cell_8_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_8/BiasAdd/ReadVariableOpв!lstm_cell_8/MatMul/ReadVariableOpв#lstm_cell_8/MatMul_1/ReadVariableOpвwhileD
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
value	B :@2
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
value	B :@2
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
:         @2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:         @2	
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
:         2
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
valueB"       27
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
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpк
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul╕
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpж
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/MatMul_1Ь
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/add▒
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpй
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_8/splitГ
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/SigmoidЗ
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_1И
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_8/ReluШ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_1Н
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/add_1З
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/Relu_1Ь
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_8/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_101558*
condR
while_cond_101557*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
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
:         @*
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
:         @2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         @2

Identity┼
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┐%
▄
while_body_100314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_100338_0:	А-
while_lstm_cell_8_100340_0:	@А)
while_lstm_cell_8_100342_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_100338:	А+
while_lstm_cell_8_100340:	@А'
while_lstm_cell_8_100342:	АИв)while/lstm_cell_8/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem█
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_100338_0while_lstm_cell_8_100340_0while_lstm_cell_8_100342_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1003002+
)while/lstm_cell_8/StatefulPartitionedCallЎ
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5Ж

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
while_lstm_cell_8_100338while_lstm_cell_8_100338_0"6
while_lstm_cell_8_100340while_lstm_cell_8_100340_0"6
while_lstm_cell_8_100342while_lstm_cell_8_100342_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2V
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_101557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_101557___redundant_placeholder04
0while_while_cond_101557___redundant_placeholder14
0while_while_cond_101557___redundant_placeholder24
0while_while_cond_101557___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
ЬF
Б
B__inference_lstm_9_layer_call_and_return_conditional_losses_101013

inputs%
lstm_cell_9_100931:	@А&
lstm_cell_9_100933:
АА!
lstm_cell_9_100935:	А
identityИв#lstm_cell_9/StatefulPartitionedCallвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
 :                  @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2Ъ
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_100931lstm_cell_9_100933lstm_cell_9_100935*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1009302%
#lstm_cell_9/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counter┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_100931lstm_cell_9_100933lstm_cell_9_100935*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100944*
condR
while_cond_100943*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity|
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
┴H
з

lstm_8_while_body_102952*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	АN
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	АL
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	@АG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	АИв/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpв.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpв0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp╤
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem█
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpЁ
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_8/while/lstm_cell_8/MatMulс
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp┘
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_8/while/lstm_cell_8/MatMul_1╨
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_8/while/lstm_cell_8/add┌
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp▌
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_8/while/lstm_cell_8/BiasAddЦ
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dimг
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2 
lstm_8/while/lstm_cell_8/splitк
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2"
 lstm_8/while/lstm_cell_8/Sigmoidо
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2$
"lstm_8/while/lstm_cell_8/Sigmoid_1╣
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:         @2
lstm_8/while/lstm_cell_8/mulб
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_8/while/lstm_cell_8/Relu╠
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/mul_1┴
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/add_1о
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2$
"lstm_8/while/lstm_cell_8/Sigmoid_2а
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2!
lstm_8/while/lstm_cell_8/Relu_1╨
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2 
lstm_8/while/lstm_cell_8/mul_2В
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/yЕ
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
lstm_8/while/add_1/yЩ
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1З
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identityб
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1Й
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2╢
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3и
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_8/while/Identity_4и
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_8/while/Identity_5■
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
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
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"─
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
нь
Ш	
H__inference_sequential_2_layer_call_and_return_conditional_losses_102885

inputsD
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	АF
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	@АA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	АD
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	@АG
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
ААA
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	А:
&dense_8_matmul_readvariableop_resource:
АА6
'dense_8_biasadd_readvariableop_resource:	А9
&dense_9_matmul_readvariableop_resource:	А5
'dense_9_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpв(lstm_8/lstm_cell_8/MatMul/ReadVariableOpв*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpвlstm_8/whileв)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpв(lstm_9/lstm_cell_9/MatMul/ReadVariableOpв*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpвlstm_9/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/ShapeВ
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stackЖ
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1Ж
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2М
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
value	B :@2
lstm_8/zeros/mul/yИ
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
lstm_8/zeros/Less/yГ
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros/packed/1Я
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
lstm_8/zeros/ConstС
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros_1/mul/yО
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
lstm_8/zeros_1/Less/yЛ
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros_1/packed/1е
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
lstm_8/zeros_1/ConstЩ
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_8/zeros_1Г
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/permП
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1Ж
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stackК
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1К
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2Ш
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1У
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_8/TensorArrayV2/element_shape╬
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2═
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensorЖ
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stackК
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1К
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2ж
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_8/strided_slice_2╟
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp╞
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/MatMul═
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp┬
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/MatMul_1╕
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/add╞
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp┼
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/BiasAddК
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dimЛ
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_8/lstm_cell_8/splitШ
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/SigmoidЬ
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Sigmoid_1д
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mulП
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Relu┤
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mul_1й
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/add_1Ь
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Sigmoid_2О
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Relu_1╕
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mul_2Э
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_8/TensorArrayV2_1/element_shape╘
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
lstm_8/timeН
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterё
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_8_while_body_102630*$
condR
lstm_8_while_cond_102629*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
lstm_8/while├
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStackП
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_8/strided_slice_3/stackК
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1К
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2─
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_8/strided_slice_3З
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm┴
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimeВ
dropout_4/IdentityIdentitylstm_8/transpose_1:y:0*
T0*+
_output_shapes
:         @2
dropout_4/Identityg
lstm_9/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:2
lstm_9/ShapeВ
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stackЖ
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1Ж
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2М
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicek
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros/mul/yИ
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_9/zeros/Less/yГ
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessq
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros/packed/1Я
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/ConstТ
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:         А2
lstm_9/zeroso
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros_1/mul/yО
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_9/zeros_1/Less/yЛ
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lessu
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros_1/packed/1е
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/ConstЪ
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2
lstm_9/zeros_1Г
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/permд
lstm_9/transpose	Transposedropout_4/Identity:output:0lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1Ж
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stackК
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1К
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2Ш
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1У
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_9/TensorArrayV2/element_shape╬
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2═
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensorЖ
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stackК
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1К
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2ж
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_9/strided_slice_2╟
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp╞
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/MatMul╬
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp┬
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/MatMul_1╕
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/add╞
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp┼
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/BiasAddК
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_9/lstm_cell_9/split/split_dimП
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_9/lstm_cell_9/splitЩ
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/SigmoidЭ
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Sigmoid_1е
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mulР
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Relu╡
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mul_1к
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/add_1Э
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Sigmoid_2П
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Relu_1╣
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mul_2Э
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2&
$lstm_9/TensorArrayV2_1/element_shape╘
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/timeН
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counterї
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_102778*$
condR
lstm_9_while_cond_102777*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
lstm_9/while├
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStackП
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_9/strided_slice_3/stackК
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1К
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2┼
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
lstm_9/strided_slice_3З
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/perm┬
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtimeИ
dropout_5/IdentityIdentitylstm_9/strided_slice_3:output:0*
T0*(
_output_shapes
:         А2
dropout_5/Identityз
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_8/MatMul/ReadVariableOpб
dense_8/MatMulMatMuldropout_5/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_8/MatMulе
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_8/BiasAdd/ReadVariableOpв
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_8/Reluж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_9/MatMul/ReadVariableOpЯ
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddj
reshape_4/ShapeShapedense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_4/ShapeИ
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stackМ
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1М
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2Ю
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2╥
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shapeг
reshape_4/ReshapeReshapedense_9/BiasAdd:output:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_4/Reshapey
IdentityIdentityreshape_4/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЎ
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Р
Ж
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104792

inputs
states_0
states_11
matmul_readvariableop_resource:	@А4
 matmul_1_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         А2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         А2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         А2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         А2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         А2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         А2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         А2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         А2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:RN
(
_output_shapes
:         А
"
_user_specified_name
states/0:RN
(
_output_shapes
:         А
"
_user_specified_name
states/1
Р
Ж
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104824

inputs
states_0
states_11
matmul_readvariableop_resource:	@А4
 matmul_1_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         А2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         А2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         А2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         А2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         А2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         А2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         А2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         А2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         А2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         А2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:RN
(
_output_shapes
:         А
"
_user_specified_name
states/0:RN
(
_output_shapes
:         А
"
_user_specified_name
states/1
╡[
Х
B__inference_lstm_9_layer_call_and_return_conditional_losses_102107

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	@А@
,lstm_cell_9_matmul_1_readvariableop_resource:
АА:
+lstm_cell_9_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_9/BiasAdd/ReadVariableOpв!lstm_cell_9/MatMul/ReadVariableOpв#lstm_cell_9/MatMul_1/ReadVariableOpвwhileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         А2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2	
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
:         @2
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
valueB"    @   27
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
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpк
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul╣
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOpж
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/MatMul_1Ь
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add▒
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOpй
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimє
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_cell_9/splitД
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/SigmoidИ
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_1Й
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_cell_9/ReluЩ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_1О
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/add_1И
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/Relu_1Э
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_cell_9/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2
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
while/loop_counterМ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_102023*
condR
while_cond_102022*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
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
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identity┼
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
┴
Ў
,__inference_lstm_cell_9_layer_call_fn_104760

inputs
states_0
states_1
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1010762
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         А2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:RN
(
_output_shapes
:         А
"
_user_specified_name
states/0:RN
(
_output_shapes
:         А
"
_user_specified_name
states/1
ЭR
Р
__inference__traced_save_104958
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop8
4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableopB
>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop6
2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop?
;savev2_adam_lstm_9_lstm_cell_9_kernel_m_read_readvariableopI
Esavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_9_lstm_cell_9_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop?
;savev2_adam_lstm_9_lstm_cell_9_kernel_v_read_readvariableopI
Esavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_9_lstm_cell_9_bias_v_read_readvariableop
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
ShardedFilenameЖ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Ш
valueОBЛ&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesя
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableop>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop;savev2_adam_lstm_9_lstm_cell_9_kernel_m_read_readvariableopEsavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_9_lstm_cell_9_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop;savev2_adam_lstm_9_lstm_cell_9_kernel_v_read_readvariableopEsavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_9_lstm_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
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

identity_1Identity_1:output:0*─
_input_shapes▓
п: :
АА:А:	А:: : : : : :	А:	@А:А:	@А:
АА:А: : :
АА:А:	А::	А:	@А:А:	@А:
АА:А:
АА:А:	А::	А:	@А:А:	@А:
АА:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 
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
:	А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::%!

_output_shapes
:	А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::% !

_output_shapes
:	А:%!!

_output_shapes
:	@А:!"

_output_shapes	
:А:%#!

_output_shapes
:	@А:&$"
 
_output_shapes
:
АА:!%

_output_shapes	
:А:&

_output_shapes
: 
┴
Ў
,__inference_lstm_cell_9_layer_call_fn_104743

inputs
states_0
states_1
unknown:	@А
	unknown_0:
АА
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_1009302
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         А2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         @:         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:RN
(
_output_shapes
:         А
"
_user_specified_name
states/0:RN
(
_output_shapes
:         А
"
_user_specified_name
states/1
И
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_104628

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
┬
F
*__inference_dropout_5_layer_call_fn_104549

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1018202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·

З
-__inference_sequential_2_layer_call_fn_102418
lstm_8_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1023702
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
┤
ї
,__inference_lstm_cell_8_layer_call_fn_104645

inputs
states_0
states_1
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1003002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @2

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
?:         :         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
·

З
-__inference_sequential_2_layer_call_fn_101894
lstm_8_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1018712
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
м
┤
'__inference_lstm_8_layer_call_fn_103265

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1023032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Рж
Х
!__inference__wrapped_model_100225
lstm_8_inputQ
>sequential_2_lstm_8_lstm_cell_8_matmul_readvariableop_resource:	АS
@sequential_2_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	@АN
?sequential_2_lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	АQ
>sequential_2_lstm_9_lstm_cell_9_matmul_readvariableop_resource:	@АT
@sequential_2_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
ААN
?sequential_2_lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	АG
3sequential_2_dense_8_matmul_readvariableop_resource:
ААC
4sequential_2_dense_8_biasadd_readvariableop_resource:	АF
3sequential_2_dense_9_matmul_readvariableop_resource:	АB
4sequential_2_dense_9_biasadd_readvariableop_resource:
identityИв+sequential_2/dense_8/BiasAdd/ReadVariableOpв*sequential_2/dense_8/MatMul/ReadVariableOpв+sequential_2/dense_9/BiasAdd/ReadVariableOpв*sequential_2/dense_9/MatMul/ReadVariableOpв6sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpв5sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOpв7sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpвsequential_2/lstm_8/whileв6sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpв5sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOpв7sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpвsequential_2/lstm_9/whiler
sequential_2/lstm_8/ShapeShapelstm_8_input*
T0*
_output_shapes
:2
sequential_2/lstm_8/ShapeЬ
'sequential_2/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/lstm_8/strided_slice/stackа
)sequential_2/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_8/strided_slice/stack_1а
)sequential_2/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_8/strided_slice/stack_2┌
!sequential_2/lstm_8/strided_sliceStridedSlice"sequential_2/lstm_8/Shape:output:00sequential_2/lstm_8/strided_slice/stack:output:02sequential_2/lstm_8/strided_slice/stack_1:output:02sequential_2/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_2/lstm_8/strided_sliceД
sequential_2/lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2!
sequential_2/lstm_8/zeros/mul/y╝
sequential_2/lstm_8/zeros/mulMul*sequential_2/lstm_8/strided_slice:output:0(sequential_2/lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_8/zeros/mulЗ
 sequential_2/lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_2/lstm_8/zeros/Less/y╖
sequential_2/lstm_8/zeros/LessLess!sequential_2/lstm_8/zeros/mul:z:0)sequential_2/lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_2/lstm_8/zeros/LessК
"sequential_2/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_2/lstm_8/zeros/packed/1╙
 sequential_2/lstm_8/zeros/packedPack*sequential_2/lstm_8/strided_slice:output:0+sequential_2/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_2/lstm_8/zeros/packedЗ
sequential_2/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_2/lstm_8/zeros/Const┼
sequential_2/lstm_8/zerosFill)sequential_2/lstm_8/zeros/packed:output:0(sequential_2/lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
sequential_2/lstm_8/zerosИ
!sequential_2/lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_2/lstm_8/zeros_1/mul/y┬
sequential_2/lstm_8/zeros_1/mulMul*sequential_2/lstm_8/strided_slice:output:0*sequential_2/lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_8/zeros_1/mulЛ
"sequential_2/lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_2/lstm_8/zeros_1/Less/y┐
 sequential_2/lstm_8/zeros_1/LessLess#sequential_2/lstm_8/zeros_1/mul:z:0+sequential_2/lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_2/lstm_8/zeros_1/LessО
$sequential_2/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_2/lstm_8/zeros_1/packed/1┘
"sequential_2/lstm_8/zeros_1/packedPack*sequential_2/lstm_8/strided_slice:output:0-sequential_2/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_2/lstm_8/zeros_1/packedЛ
!sequential_2/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_2/lstm_8/zeros_1/Const═
sequential_2/lstm_8/zeros_1Fill+sequential_2/lstm_8/zeros_1/packed:output:0*sequential_2/lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
sequential_2/lstm_8/zeros_1Э
"sequential_2/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_2/lstm_8/transpose/perm╝
sequential_2/lstm_8/transpose	Transposelstm_8_input+sequential_2/lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:         2
sequential_2/lstm_8/transposeЛ
sequential_2/lstm_8/Shape_1Shape!sequential_2/lstm_8/transpose:y:0*
T0*
_output_shapes
:2
sequential_2/lstm_8/Shape_1а
)sequential_2/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_8/strided_slice_1/stackд
+sequential_2/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_8/strided_slice_1/stack_1д
+sequential_2/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_8/strided_slice_1/stack_2ц
#sequential_2/lstm_8/strided_slice_1StridedSlice$sequential_2/lstm_8/Shape_1:output:02sequential_2/lstm_8/strided_slice_1/stack:output:04sequential_2/lstm_8/strided_slice_1/stack_1:output:04sequential_2/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_2/lstm_8/strided_slice_1н
/sequential_2/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         21
/sequential_2/lstm_8/TensorArrayV2/element_shapeВ
!sequential_2/lstm_8/TensorArrayV2TensorListReserve8sequential_2/lstm_8/TensorArrayV2/element_shape:output:0,sequential_2/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_2/lstm_8/TensorArrayV2ч
Isequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2K
Isequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape╚
;sequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_2/lstm_8/transpose:y:0Rsequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensorа
)sequential_2/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_8/strided_slice_2/stackд
+sequential_2/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_8/strided_slice_2/stack_1д
+sequential_2/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_8/strided_slice_2/stack_2Ї
#sequential_2/lstm_8/strided_slice_2StridedSlice!sequential_2/lstm_8/transpose:y:02sequential_2/lstm_8/strided_slice_2/stack:output:04sequential_2/lstm_8/strided_slice_2/stack_1:output:04sequential_2/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2%
#sequential_2/lstm_8/strided_slice_2ю
5sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp>sequential_2_lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype027
5sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOp·
&sequential_2/lstm_8/lstm_cell_8/MatMulMatMul,sequential_2/lstm_8/strided_slice_2:output:0=sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2(
&sequential_2/lstm_8/lstm_cell_8/MatMulЇ
7sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp@sequential_2_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype029
7sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpЎ
(sequential_2/lstm_8/lstm_cell_8/MatMul_1MatMul"sequential_2/lstm_8/zeros:output:0?sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(sequential_2/lstm_8/lstm_cell_8/MatMul_1ь
#sequential_2/lstm_8/lstm_cell_8/addAddV20sequential_2/lstm_8/lstm_cell_8/MatMul:product:02sequential_2/lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2%
#sequential_2/lstm_8/lstm_cell_8/addэ
6sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp∙
'sequential_2/lstm_8/lstm_cell_8/BiasAddBiasAdd'sequential_2/lstm_8/lstm_cell_8/add:z:0>sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2)
'sequential_2/lstm_8/lstm_cell_8/BiasAddд
/sequential_2/lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/lstm_8/lstm_cell_8/split/split_dim┐
%sequential_2/lstm_8/lstm_cell_8/splitSplit8sequential_2/lstm_8/lstm_cell_8/split/split_dim:output:00sequential_2/lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2'
%sequential_2/lstm_8/lstm_cell_8/split┐
'sequential_2/lstm_8/lstm_cell_8/SigmoidSigmoid.sequential_2/lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2)
'sequential_2/lstm_8/lstm_cell_8/Sigmoid├
)sequential_2/lstm_8/lstm_cell_8/Sigmoid_1Sigmoid.sequential_2/lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2+
)sequential_2/lstm_8/lstm_cell_8/Sigmoid_1╪
#sequential_2/lstm_8/lstm_cell_8/mulMul-sequential_2/lstm_8/lstm_cell_8/Sigmoid_1:y:0$sequential_2/lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:         @2%
#sequential_2/lstm_8/lstm_cell_8/mul╢
$sequential_2/lstm_8/lstm_cell_8/ReluRelu.sequential_2/lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2&
$sequential_2/lstm_8/lstm_cell_8/Reluш
%sequential_2/lstm_8/lstm_cell_8/mul_1Mul+sequential_2/lstm_8/lstm_cell_8/Sigmoid:y:02sequential_2/lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2'
%sequential_2/lstm_8/lstm_cell_8/mul_1▌
%sequential_2/lstm_8/lstm_cell_8/add_1AddV2'sequential_2/lstm_8/lstm_cell_8/mul:z:0)sequential_2/lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2'
%sequential_2/lstm_8/lstm_cell_8/add_1├
)sequential_2/lstm_8/lstm_cell_8/Sigmoid_2Sigmoid.sequential_2/lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2+
)sequential_2/lstm_8/lstm_cell_8/Sigmoid_2╡
&sequential_2/lstm_8/lstm_cell_8/Relu_1Relu)sequential_2/lstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2(
&sequential_2/lstm_8/lstm_cell_8/Relu_1ь
%sequential_2/lstm_8/lstm_cell_8/mul_2Mul-sequential_2/lstm_8/lstm_cell_8/Sigmoid_2:y:04sequential_2/lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2'
%sequential_2/lstm_8/lstm_cell_8/mul_2╖
1sequential_2/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   23
1sequential_2/lstm_8/TensorArrayV2_1/element_shapeИ
#sequential_2/lstm_8/TensorArrayV2_1TensorListReserve:sequential_2/lstm_8/TensorArrayV2_1/element_shape:output:0,sequential_2/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_2/lstm_8/TensorArrayV2_1v
sequential_2/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/lstm_8/timeз
,sequential_2/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,sequential_2/lstm_8/while/maximum_iterationsТ
&sequential_2/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_2/lstm_8/while/loop_counter▓
sequential_2/lstm_8/whileWhile/sequential_2/lstm_8/while/loop_counter:output:05sequential_2/lstm_8/while/maximum_iterations:output:0!sequential_2/lstm_8/time:output:0,sequential_2/lstm_8/TensorArrayV2_1:handle:0"sequential_2/lstm_8/zeros:output:0$sequential_2/lstm_8/zeros_1:output:0,sequential_2/lstm_8/strided_slice_1:output:0Ksequential_2/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_2_lstm_8_lstm_cell_8_matmul_readvariableop_resource@sequential_2_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource?sequential_2_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_2_lstm_8_while_body_99970*0
cond(R&
$sequential_2_lstm_8_while_cond_99969*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
sequential_2/lstm_8/while▌
Dsequential_2/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2F
Dsequential_2/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape╕
6sequential_2/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_2/lstm_8/while:output:3Msequential_2/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype028
6sequential_2/lstm_8/TensorArrayV2Stack/TensorListStackй
)sequential_2/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2+
)sequential_2/lstm_8/strided_slice_3/stackд
+sequential_2/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_2/lstm_8/strided_slice_3/stack_1д
+sequential_2/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_8/strided_slice_3/stack_2Т
#sequential_2/lstm_8/strided_slice_3StridedSlice?sequential_2/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:02sequential_2/lstm_8/strided_slice_3/stack:output:04sequential_2/lstm_8/strided_slice_3/stack_1:output:04sequential_2/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2%
#sequential_2/lstm_8/strided_slice_3б
$sequential_2/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_2/lstm_8/transpose_1/permї
sequential_2/lstm_8/transpose_1	Transpose?sequential_2/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_2/lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2!
sequential_2/lstm_8/transpose_1О
sequential_2/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_2/lstm_8/runtimeй
sequential_2/dropout_4/IdentityIdentity#sequential_2/lstm_8/transpose_1:y:0*
T0*+
_output_shapes
:         @2!
sequential_2/dropout_4/IdentityО
sequential_2/lstm_9/ShapeShape(sequential_2/dropout_4/Identity:output:0*
T0*
_output_shapes
:2
sequential_2/lstm_9/ShapeЬ
'sequential_2/lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/lstm_9/strided_slice/stackа
)sequential_2/lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_9/strided_slice/stack_1а
)sequential_2/lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_2/lstm_9/strided_slice/stack_2┌
!sequential_2/lstm_9/strided_sliceStridedSlice"sequential_2/lstm_9/Shape:output:00sequential_2/lstm_9/strided_slice/stack:output:02sequential_2/lstm_9/strided_slice/stack_1:output:02sequential_2/lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_2/lstm_9/strided_sliceЕ
sequential_2/lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2!
sequential_2/lstm_9/zeros/mul/y╝
sequential_2/lstm_9/zeros/mulMul*sequential_2/lstm_9/strided_slice:output:0(sequential_2/lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_9/zeros/mulЗ
 sequential_2/lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_2/lstm_9/zeros/Less/y╖
sequential_2/lstm_9/zeros/LessLess!sequential_2/lstm_9/zeros/mul:z:0)sequential_2/lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_2/lstm_9/zeros/LessЛ
"sequential_2/lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2$
"sequential_2/lstm_9/zeros/packed/1╙
 sequential_2/lstm_9/zeros/packedPack*sequential_2/lstm_9/strided_slice:output:0+sequential_2/lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_2/lstm_9/zeros/packedЗ
sequential_2/lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_2/lstm_9/zeros/Const╞
sequential_2/lstm_9/zerosFill)sequential_2/lstm_9/zeros/packed:output:0(sequential_2/lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:         А2
sequential_2/lstm_9/zerosЙ
!sequential_2/lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2#
!sequential_2/lstm_9/zeros_1/mul/y┬
sequential_2/lstm_9/zeros_1/mulMul*sequential_2/lstm_9/strided_slice:output:0*sequential_2/lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_9/zeros_1/mulЛ
"sequential_2/lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_2/lstm_9/zeros_1/Less/y┐
 sequential_2/lstm_9/zeros_1/LessLess#sequential_2/lstm_9/zeros_1/mul:z:0+sequential_2/lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_2/lstm_9/zeros_1/LessП
$sequential_2/lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2&
$sequential_2/lstm_9/zeros_1/packed/1┘
"sequential_2/lstm_9/zeros_1/packedPack*sequential_2/lstm_9/strided_slice:output:0-sequential_2/lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_2/lstm_9/zeros_1/packedЛ
!sequential_2/lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_2/lstm_9/zeros_1/Const╬
sequential_2/lstm_9/zeros_1Fill+sequential_2/lstm_9/zeros_1/packed:output:0*sequential_2/lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2
sequential_2/lstm_9/zeros_1Э
"sequential_2/lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_2/lstm_9/transpose/perm╪
sequential_2/lstm_9/transpose	Transpose(sequential_2/dropout_4/Identity:output:0+sequential_2/lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
sequential_2/lstm_9/transposeЛ
sequential_2/lstm_9/Shape_1Shape!sequential_2/lstm_9/transpose:y:0*
T0*
_output_shapes
:2
sequential_2/lstm_9/Shape_1а
)sequential_2/lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_9/strided_slice_1/stackд
+sequential_2/lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_9/strided_slice_1/stack_1д
+sequential_2/lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_9/strided_slice_1/stack_2ц
#sequential_2/lstm_9/strided_slice_1StridedSlice$sequential_2/lstm_9/Shape_1:output:02sequential_2/lstm_9/strided_slice_1/stack:output:04sequential_2/lstm_9/strided_slice_1/stack_1:output:04sequential_2/lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_2/lstm_9/strided_slice_1н
/sequential_2/lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         21
/sequential_2/lstm_9/TensorArrayV2/element_shapeВ
!sequential_2/lstm_9/TensorArrayV2TensorListReserve8sequential_2/lstm_9/TensorArrayV2/element_shape:output:0,sequential_2/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_2/lstm_9/TensorArrayV2ч
Isequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2K
Isequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape╚
;sequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_2/lstm_9/transpose:y:0Rsequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensorа
)sequential_2/lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_2/lstm_9/strided_slice_2/stackд
+sequential_2/lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_9/strided_slice_2/stack_1д
+sequential_2/lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_9/strided_slice_2/stack_2Ї
#sequential_2/lstm_9/strided_slice_2StridedSlice!sequential_2/lstm_9/transpose:y:02sequential_2/lstm_9/strided_slice_2/stack:output:04sequential_2/lstm_9/strided_slice_2/stack_1:output:04sequential_2/lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2%
#sequential_2/lstm_9/strided_slice_2ю
5sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp>sequential_2_lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype027
5sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOp·
&sequential_2/lstm_9/lstm_cell_9/MatMulMatMul,sequential_2/lstm_9/strided_slice_2:output:0=sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2(
&sequential_2/lstm_9/lstm_cell_9/MatMulї
7sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp@sequential_2_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype029
7sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpЎ
(sequential_2/lstm_9/lstm_cell_9/MatMul_1MatMul"sequential_2/lstm_9/zeros:output:0?sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(sequential_2/lstm_9/lstm_cell_9/MatMul_1ь
#sequential_2/lstm_9/lstm_cell_9/addAddV20sequential_2/lstm_9/lstm_cell_9/MatMul:product:02sequential_2/lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2%
#sequential_2/lstm_9/lstm_cell_9/addэ
6sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp∙
'sequential_2/lstm_9/lstm_cell_9/BiasAddBiasAdd'sequential_2/lstm_9/lstm_cell_9/add:z:0>sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2)
'sequential_2/lstm_9/lstm_cell_9/BiasAddд
/sequential_2/lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/lstm_9/lstm_cell_9/split/split_dim├
%sequential_2/lstm_9/lstm_cell_9/splitSplit8sequential_2/lstm_9/lstm_cell_9/split/split_dim:output:00sequential_2/lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2'
%sequential_2/lstm_9/lstm_cell_9/split└
'sequential_2/lstm_9/lstm_cell_9/SigmoidSigmoid.sequential_2/lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2)
'sequential_2/lstm_9/lstm_cell_9/Sigmoid─
)sequential_2/lstm_9/lstm_cell_9/Sigmoid_1Sigmoid.sequential_2/lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2+
)sequential_2/lstm_9/lstm_cell_9/Sigmoid_1┘
#sequential_2/lstm_9/lstm_cell_9/mulMul-sequential_2/lstm_9/lstm_cell_9/Sigmoid_1:y:0$sequential_2/lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:         А2%
#sequential_2/lstm_9/lstm_cell_9/mul╖
$sequential_2/lstm_9/lstm_cell_9/ReluRelu.sequential_2/lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2&
$sequential_2/lstm_9/lstm_cell_9/Reluщ
%sequential_2/lstm_9/lstm_cell_9/mul_1Mul+sequential_2/lstm_9/lstm_cell_9/Sigmoid:y:02sequential_2/lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2'
%sequential_2/lstm_9/lstm_cell_9/mul_1▐
%sequential_2/lstm_9/lstm_cell_9/add_1AddV2'sequential_2/lstm_9/lstm_cell_9/mul:z:0)sequential_2/lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2'
%sequential_2/lstm_9/lstm_cell_9/add_1─
)sequential_2/lstm_9/lstm_cell_9/Sigmoid_2Sigmoid.sequential_2/lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2+
)sequential_2/lstm_9/lstm_cell_9/Sigmoid_2╢
&sequential_2/lstm_9/lstm_cell_9/Relu_1Relu)sequential_2/lstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2(
&sequential_2/lstm_9/lstm_cell_9/Relu_1э
%sequential_2/lstm_9/lstm_cell_9/mul_2Mul-sequential_2/lstm_9/lstm_cell_9/Sigmoid_2:y:04sequential_2/lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2'
%sequential_2/lstm_9/lstm_cell_9/mul_2╖
1sequential_2/lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   23
1sequential_2/lstm_9/TensorArrayV2_1/element_shapeИ
#sequential_2/lstm_9/TensorArrayV2_1TensorListReserve:sequential_2/lstm_9/TensorArrayV2_1/element_shape:output:0,sequential_2/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_2/lstm_9/TensorArrayV2_1v
sequential_2/lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/lstm_9/timeз
,sequential_2/lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,sequential_2/lstm_9/while/maximum_iterationsТ
&sequential_2/lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_2/lstm_9/while/loop_counter╕
sequential_2/lstm_9/whileWhile/sequential_2/lstm_9/while/loop_counter:output:05sequential_2/lstm_9/while/maximum_iterations:output:0!sequential_2/lstm_9/time:output:0,sequential_2/lstm_9/TensorArrayV2_1:handle:0"sequential_2/lstm_9/zeros:output:0$sequential_2/lstm_9/zeros_1:output:0,sequential_2/lstm_9/strided_slice_1:output:0Ksequential_2/lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_2_lstm_9_lstm_cell_9_matmul_readvariableop_resource@sequential_2_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource?sequential_2_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_2_lstm_9_while_body_100118*1
cond)R'
%sequential_2_lstm_9_while_cond_100117*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
sequential_2/lstm_9/while▌
Dsequential_2/lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2F
Dsequential_2/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape╣
6sequential_2/lstm_9/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_2/lstm_9/while:output:3Msequential_2/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype028
6sequential_2/lstm_9/TensorArrayV2Stack/TensorListStackй
)sequential_2/lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2+
)sequential_2/lstm_9/strided_slice_3/stackд
+sequential_2/lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_2/lstm_9/strided_slice_3/stack_1д
+sequential_2/lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_2/lstm_9/strided_slice_3/stack_2У
#sequential_2/lstm_9/strided_slice_3StridedSlice?sequential_2/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:02sequential_2/lstm_9/strided_slice_3/stack:output:04sequential_2/lstm_9/strided_slice_3/stack_1:output:04sequential_2/lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2%
#sequential_2/lstm_9/strided_slice_3б
$sequential_2/lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_2/lstm_9/transpose_1/permЎ
sequential_2/lstm_9/transpose_1	Transpose?sequential_2/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_2/lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2!
sequential_2/lstm_9/transpose_1О
sequential_2/lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_2/lstm_9/runtimeп
sequential_2/dropout_5/IdentityIdentity,sequential_2/lstm_9/strided_slice_3:output:0*
T0*(
_output_shapes
:         А2!
sequential_2/dropout_5/Identity╬
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp╒
sequential_2/dense_8/MatMulMatMul(sequential_2/dropout_5/Identity:output:02sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_2/dense_8/MatMul╠
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOp╓
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_2/dense_8/BiasAddШ
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_2/dense_8/Relu═
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOp╙
sequential_2/dense_9/MatMulMatMul'sequential_2/dense_8/Relu:activations:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_2/dense_9/MatMul╦
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp╒
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_2/dense_9/BiasAddС
sequential_2/reshape_4/ShapeShape%sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_2/reshape_4/Shapeв
*sequential_2/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_2/reshape_4/strided_slice/stackж
,sequential_2/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_4/strided_slice/stack_1ж
,sequential_2/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_4/strided_slice/stack_2ь
$sequential_2/reshape_4/strided_sliceStridedSlice%sequential_2/reshape_4/Shape:output:03sequential_2/reshape_4/strided_slice/stack:output:05sequential_2/reshape_4/strided_slice/stack_1:output:05sequential_2/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_2/reshape_4/strided_sliceТ
&sequential_2/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_4/Reshape/shape/1Т
&sequential_2/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_4/Reshape/shape/2У
$sequential_2/reshape_4/Reshape/shapePack-sequential_2/reshape_4/strided_slice:output:0/sequential_2/reshape_4/Reshape/shape/1:output:0/sequential_2/reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/reshape_4/Reshape/shape╫
sequential_2/reshape_4/ReshapeReshape%sequential_2/dense_9/BiasAdd:output:0-sequential_2/reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:         2 
sequential_2/reshape_4/ReshapeЖ
IdentityIdentity'sequential_2/reshape_4/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityТ
NoOpNoOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp7^sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6^sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOp8^sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^sequential_2/lstm_8/while7^sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6^sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOp8^sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^sequential_2/lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2p
6sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6sequential_2/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2n
5sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOp5sequential_2/lstm_8/lstm_cell_8/MatMul/ReadVariableOp2r
7sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp7sequential_2/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp26
sequential_2/lstm_8/whilesequential_2/lstm_8/while2p
6sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6sequential_2/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2n
5sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOp5sequential_2/lstm_9/lstm_cell_9/MatMul/ReadVariableOp2r
7sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp7sequential_2/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp26
sequential_2/lstm_9/whilesequential_2/lstm_9/while:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_8_input
╒
├
while_cond_103482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103482___redundant_placeholder04
0while_while_cond_103482___redundant_placeholder14
0while_while_cond_103482___redundant_placeholder24
0while_while_cond_103482___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╪>
╔
while_body_104460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
╪>
╔
while_body_102023
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	@АH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	@АF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
АА@
1while_lstm_cell_9_biasadd_readvariableop_resource:	АИв(while/lstm_cell_9/BiasAdd/ReadVariableOpв'while/lstm_cell_9/MatMul/ReadVariableOpв)while/lstm_cell_9/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOp╘
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul═
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp╜
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/MatMul_1┤
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add┼
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOp┴
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/BiasAddИ
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dimЛ
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
while/lstm_cell_9/splitЦ
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/SigmoidЪ
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_1Ю
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mulН
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu▒
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_1ж
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/add_1Ъ
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Sigmoid_2М
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/Relu_1╡
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_9/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         А2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
м

╧
lstm_9_while_cond_103106*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_103106___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_103106___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_103106___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_103106___redundant_placeholder3
lstm_9_while_identity
У
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: ::::: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
:
Ы[
╔
%sequential_2_lstm_9_while_body_100118D
@sequential_2_lstm_9_while_sequential_2_lstm_9_while_loop_counterJ
Fsequential_2_lstm_9_while_sequential_2_lstm_9_while_maximum_iterations)
%sequential_2_lstm_9_while_placeholder+
'sequential_2_lstm_9_while_placeholder_1+
'sequential_2_lstm_9_while_placeholder_2+
'sequential_2_lstm_9_while_placeholder_3C
?sequential_2_lstm_9_while_sequential_2_lstm_9_strided_slice_1_0
{sequential_2_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_9_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_2_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	@А\
Hsequential_2_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
ААV
Gsequential_2_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	А&
"sequential_2_lstm_9_while_identity(
$sequential_2_lstm_9_while_identity_1(
$sequential_2_lstm_9_while_identity_2(
$sequential_2_lstm_9_while_identity_3(
$sequential_2_lstm_9_while_identity_4(
$sequential_2_lstm_9_while_identity_5A
=sequential_2_lstm_9_while_sequential_2_lstm_9_strided_slice_1}
ysequential_2_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_9_tensorarrayunstack_tensorlistfromtensorW
Dsequential_2_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	@АZ
Fsequential_2_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
ААT
Esequential_2_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	АИв<sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpв;sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpв=sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpы
Ksequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2M
Ksequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape╦
=sequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_2_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_9_tensorarrayunstack_tensorlistfromtensor_0%sequential_2_lstm_9_while_placeholderTsequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02?
=sequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItemВ
;sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOpFsequential_2_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02=
;sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpд
,sequential_2/lstm_9/while/lstm_cell_9/MatMulMatMulDsequential_2/lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,sequential_2/lstm_9/while/lstm_cell_9/MatMulЙ
=sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOpHsequential_2_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype02?
=sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpН
.sequential_2/lstm_9/while/lstm_cell_9/MatMul_1MatMul'sequential_2_lstm_9_while_placeholder_2Esequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А20
.sequential_2/lstm_9/while/lstm_cell_9/MatMul_1Д
)sequential_2/lstm_9/while/lstm_cell_9/addAddV26sequential_2/lstm_9/while/lstm_cell_9/MatMul:product:08sequential_2/lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2+
)sequential_2/lstm_9/while/lstm_cell_9/addБ
<sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOpGsequential_2_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpС
-sequential_2/lstm_9/while/lstm_cell_9/BiasAddBiasAdd-sequential_2/lstm_9/while/lstm_cell_9/add:z:0Dsequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-sequential_2/lstm_9/while/lstm_cell_9/BiasAdd░
5sequential_2/lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_2/lstm_9/while/lstm_cell_9/split/split_dim█
+sequential_2/lstm_9/while/lstm_cell_9/splitSplit>sequential_2/lstm_9/while/lstm_cell_9/split/split_dim:output:06sequential_2/lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2-
+sequential_2/lstm_9/while/lstm_cell_9/split╥
-sequential_2/lstm_9/while/lstm_cell_9/SigmoidSigmoid4sequential_2/lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2/
-sequential_2/lstm_9/while/lstm_cell_9/Sigmoid╓
/sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid4sequential_2/lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А21
/sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_1ю
)sequential_2/lstm_9/while/lstm_cell_9/mulMul3sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_1:y:0'sequential_2_lstm_9_while_placeholder_3*
T0*(
_output_shapes
:         А2+
)sequential_2/lstm_9/while/lstm_cell_9/mul╔
*sequential_2/lstm_9/while/lstm_cell_9/ReluRelu4sequential_2/lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2,
*sequential_2/lstm_9/while/lstm_cell_9/ReluБ
+sequential_2/lstm_9/while/lstm_cell_9/mul_1Mul1sequential_2/lstm_9/while/lstm_cell_9/Sigmoid:y:08sequential_2/lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2-
+sequential_2/lstm_9/while/lstm_cell_9/mul_1Ў
+sequential_2/lstm_9/while/lstm_cell_9/add_1AddV2-sequential_2/lstm_9/while/lstm_cell_9/mul:z:0/sequential_2/lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2-
+sequential_2/lstm_9/while/lstm_cell_9/add_1╓
/sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid4sequential_2/lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А21
/sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_2╚
,sequential_2/lstm_9/while/lstm_cell_9/Relu_1Relu/sequential_2/lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2.
,sequential_2/lstm_9/while/lstm_cell_9/Relu_1Е
+sequential_2/lstm_9/while/lstm_cell_9/mul_2Mul3sequential_2/lstm_9/while/lstm_cell_9/Sigmoid_2:y:0:sequential_2/lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2-
+sequential_2/lstm_9/while/lstm_cell_9/mul_2├
>sequential_2/lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_2_lstm_9_while_placeholder_1%sequential_2_lstm_9_while_placeholder/sequential_2/lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_2/lstm_9/while/TensorArrayV2Write/TensorListSetItemД
sequential_2/lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_2/lstm_9/while/add/y╣
sequential_2/lstm_9/while/addAddV2%sequential_2_lstm_9_while_placeholder(sequential_2/lstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_2/lstm_9/while/addИ
!sequential_2/lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_2/lstm_9/while/add_1/y┌
sequential_2/lstm_9/while/add_1AddV2@sequential_2_lstm_9_while_sequential_2_lstm_9_while_loop_counter*sequential_2/lstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_2/lstm_9/while/add_1╗
"sequential_2/lstm_9/while/IdentityIdentity#sequential_2/lstm_9/while/add_1:z:0^sequential_2/lstm_9/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_2/lstm_9/while/Identityт
$sequential_2/lstm_9/while/Identity_1IdentityFsequential_2_lstm_9_while_sequential_2_lstm_9_while_maximum_iterations^sequential_2/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_9/while/Identity_1╜
$sequential_2/lstm_9/while/Identity_2Identity!sequential_2/lstm_9/while/add:z:0^sequential_2/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_9/while/Identity_2ъ
$sequential_2/lstm_9/while/Identity_3IdentityNsequential_2/lstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_2/lstm_9/while/Identity_3▌
$sequential_2/lstm_9/while/Identity_4Identity/sequential_2/lstm_9/while/lstm_cell_9/mul_2:z:0^sequential_2/lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2&
$sequential_2/lstm_9/while/Identity_4▌
$sequential_2/lstm_9/while/Identity_5Identity/sequential_2/lstm_9/while/lstm_cell_9/add_1:z:0^sequential_2/lstm_9/while/NoOp*
T0*(
_output_shapes
:         А2&
$sequential_2/lstm_9/while/Identity_5┐
sequential_2/lstm_9/while/NoOpNoOp=^sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<^sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp>^sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_2/lstm_9/while/NoOp"Q
"sequential_2_lstm_9_while_identity+sequential_2/lstm_9/while/Identity:output:0"U
$sequential_2_lstm_9_while_identity_1-sequential_2/lstm_9/while/Identity_1:output:0"U
$sequential_2_lstm_9_while_identity_2-sequential_2/lstm_9/while/Identity_2:output:0"U
$sequential_2_lstm_9_while_identity_3-sequential_2/lstm_9/while/Identity_3:output:0"U
$sequential_2_lstm_9_while_identity_4-sequential_2/lstm_9/while/Identity_4:output:0"U
$sequential_2_lstm_9_while_identity_5-sequential_2/lstm_9/while/Identity_5:output:0"Р
Esequential_2_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resourceGsequential_2_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"Т
Fsequential_2_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resourceHsequential_2_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"О
Dsequential_2_lstm_9_while_lstm_cell_9_matmul_readvariableop_resourceFsequential_2_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"А
=sequential_2_lstm_9_while_sequential_2_lstm_9_strided_slice_1?sequential_2_lstm_9_while_sequential_2_lstm_9_strided_slice_1_0"°
ysequential_2_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_9_tensorarrayunstack_tensorlistfromtensor{sequential_2_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2|
<sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<sequential_2/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2z
;sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp;sequential_2/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2~
=sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp=sequential_2/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: 
д
c
*__inference_dropout_5_layer_call_fn_104554

inputs
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1019402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┤
ї
,__inference_lstm_cell_8_layer_call_fn_104662

inputs
states_0
states_1
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1004462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @2

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
?:         :         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
╒
├
while_cond_103784
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103784___redundant_placeholder04
0while_while_cond_103784___redundant_placeholder14
0while_while_cond_103784___redundant_placeholder24
0while_while_cond_103784___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
Є
Г
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_100446

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates
┬>
╟
while_body_103785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Х 
Ш	
H__inference_sequential_2_layer_call_and_return_conditional_losses_103221

inputsD
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	АF
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	@АA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	АD
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	@АG
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
ААA
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	А:
&dense_8_matmul_readvariableop_resource:
АА6
'dense_8_biasadd_readvariableop_resource:	А9
&dense_9_matmul_readvariableop_resource:	А5
'dense_9_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpв(lstm_8/lstm_cell_8/MatMul/ReadVariableOpв*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpвlstm_8/whileв)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpв(lstm_9/lstm_cell_9/MatMul/ReadVariableOpв*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpвlstm_9/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/ShapeВ
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stackЖ
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1Ж
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2М
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
value	B :@2
lstm_8/zeros/mul/yИ
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
lstm_8/zeros/Less/yГ
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros/packed/1Я
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
lstm_8/zeros/ConstС
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros_1/mul/yО
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
lstm_8/zeros_1/Less/yЛ
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_8/zeros_1/packed/1е
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
lstm_8/zeros_1/ConstЩ
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_8/zeros_1Г
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/permП
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1Ж
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stackК
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1К
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2Ш
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1У
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_8/TensorArrayV2/element_shape╬
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2═
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensorЖ
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stackК
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1К
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2ж
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_8/strided_slice_2╟
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp╞
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/MatMul═
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp┬
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/MatMul_1╕
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/add╞
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp┼
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_8/lstm_cell_8/BiasAddК
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dimЛ
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_8/lstm_cell_8/splitШ
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/SigmoidЬ
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Sigmoid_1д
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mulП
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Relu┤
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mul_1й
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/add_1Ь
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Sigmoid_2О
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/Relu_1╕
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_8/lstm_cell_8/mul_2Э
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_8/TensorArrayV2_1/element_shape╘
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
lstm_8/timeН
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterё
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_8_while_body_102952*$
condR
lstm_8_while_cond_102951*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
lstm_8/while├
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStackП
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_8/strided_slice_3/stackК
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1К
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2─
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_8/strided_slice_3З
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm┴
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_4/dropout/Constе
dropout_4/dropout/MulMullstm_8/transpose_1:y:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:         @2
dropout_4/dropout/Mulx
dropout_4/dropout/ShapeShapelstm_8/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape╓
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_4/dropout/GreaterEqual/yъ
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @2 
dropout_4/dropout/GreaterEqualб
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @2
dropout_4/dropout/Castж
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:         @2
dropout_4/dropout/Mul_1g
lstm_9/ShapeShapedropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_9/ShapeВ
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stackЖ
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1Ж
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2М
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicek
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros/mul/yИ
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_9/zeros/Less/yГ
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessq
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros/packed/1Я
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/ConstТ
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:         А2
lstm_9/zeroso
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros_1/mul/yО
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_9/zeros_1/Less/yЛ
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lessu
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А2
lstm_9/zeros_1/packed/1е
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/ConstЪ
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:         А2
lstm_9/zeros_1Г
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/permд
lstm_9/transpose	Transposedropout_4/dropout/Mul_1:z:0lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1Ж
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stackК
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1К
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2Ш
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1У
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_9/TensorArrayV2/element_shape╬
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2═
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensorЖ
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stackК
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1К
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2ж
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_9/strided_slice_2╟
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp╞
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/MatMul╬
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp┬
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/MatMul_1╕
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/add╞
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp┼
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/BiasAddК
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_9/lstm_cell_9/split/split_dimП
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split2
lstm_9/lstm_cell_9/splitЩ
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/SigmoidЭ
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Sigmoid_1е
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mulР
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Relu╡
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mul_1к
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/add_1Э
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Sigmoid_2П
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/Relu_1╣
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:         А2
lstm_9/lstm_cell_9/mul_2Э
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   2&
$lstm_9/TensorArrayV2_1/element_shape╘
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/timeН
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counterї
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_103107*$
condR
lstm_9_while_cond_103106*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations 2
lstm_9/while├
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStackП
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_9/strided_slice_3/stackК
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1К
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2┼
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask2
lstm_9/strided_slice_3З
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/perm┬
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtimew
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_5/dropout/Constл
dropout_5/dropout/MulMullstm_9/strided_slice_3:output:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_5/dropout/MulБ
dropout_5/dropout/ShapeShapelstm_9/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape╙
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype020
.dropout_5/dropout/random_uniform/RandomUniformЙ
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_5/dropout/GreaterEqual/yч
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2 
dropout_5/dropout/GreaterEqualЮ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_5/dropout/Castг
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_5/dropout/Mul_1з
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_8/MatMul/ReadVariableOpб
dense_8/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_8/MatMulе
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_8/BiasAdd/ReadVariableOpв
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_8/Reluж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_9/MatMul/ReadVariableOpЯ
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddj
reshape_4/ShapeShapedense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_4/ShapeИ
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stackМ
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1М
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2Ю
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2╥
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shapeг
reshape_4/ReshapeReshapedense_9/BiasAdd:output:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_4/Reshapey
IdentityIdentityreshape_4/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЎ
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┬>
╟
while_body_103332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	АE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_8_biasadd_readvariableop_resource:	АИв(while/lstm_cell_8/BiasAdd/ReadVariableOpв'while/lstm_cell_8/MatMul/ReadVariableOpв)while/lstm_cell_8/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp╘
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul╠
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp╜
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/MatMul_1┤
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/add┼
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp┴
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_8/BiasAddИ
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dimЗ
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_8/splitХ
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/SigmoidЩ
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_1Э
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mulМ
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu░
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_1е
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/add_1Щ
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Sigmoid_2Л
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/Relu_1┤
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_8/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Н
ў
C__inference_dense_8_layer_call_and_return_conditional_losses_101833

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╛
serving_defaultк
I
lstm_8_input9
serving_default_lstm_8_input:0         A
	reshape_44
StatefulPartitionedCall:0         tensorflow/serving/predict:Я╤
м
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
Ф_default_save_signature
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_sequential
┼
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
	variables
regularization_losses
trainable_variables
	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
	variables
regularization_losses
 trainable_variables
!	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
б__call__
+в&call_and_return_all_conditional_losses"
_tf_keras_layer
з
.	variables
/regularization_losses
0trainable_variables
1	keras_api
г__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
Ы
2iter

3beta_1

4beta_2
	5decay
6learning_rate"mА#mБ(mВ)mГ7mД8mЕ9mЖ:mЗ;mИ<mЙ"vК#vЛ(vМ)vН7vО8vП9vР:vС;vТ<vУ"
	optimizer
f
70
81
92
:3
;4
<5
"6
#7
(8
)9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
70
81
92
:3
;4
<5
"6
#7
(8
)9"
trackable_list_wrapper
╬
		variables

=layers
>non_trainable_variables
?metrics

regularization_losses
trainable_variables
@layer_regularization_losses
Alayer_metrics
Х__call__
Ф_default_save_signature
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
-
еserving_default"
signature_map
у
B
state_size

7kernel
8recurrent_kernel
9bias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
╝
	variables

Glayers
Hnon_trainable_variables
Imetrics

Jstates
regularization_losses
trainable_variables
Klayer_regularization_losses
Llayer_metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
regularization_losses
trainable_variables
Player_regularization_losses

Qlayers
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
у
R
state_size

:kernel
;recurrent_kernel
<bias
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
╝
	variables

Wlayers
Xnon_trainable_variables
Ymetrics

Zstates
regularization_losses
trainable_variables
[layer_regularization_losses
\layer_metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
]layer_metrics
^non_trainable_variables
_metrics
regularization_losses
 trainable_variables
`layer_regularization_losses

alayers
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
": 
АА2dense_8/kernel
:А2dense_8/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
░
$	variables
blayer_metrics
cnon_trainable_variables
dmetrics
%regularization_losses
&trainable_variables
elayer_regularization_losses

flayers
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_9/kernel
:2dense_9/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
░
*	variables
glayer_metrics
hnon_trainable_variables
imetrics
+regularization_losses
,trainable_variables
jlayer_regularization_losses

klayers
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
.	variables
llayer_metrics
mnon_trainable_variables
nmetrics
/regularization_losses
0trainable_variables
olayer_regularization_losses

players
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	А2lstm_8/lstm_cell_8/kernel
6:4	@А2#lstm_8/lstm_cell_8/recurrent_kernel
&:$А2lstm_8/lstm_cell_8/bias
,:*	@А2lstm_9/lstm_cell_9/kernel
7:5
АА2#lstm_9/lstm_cell_9/recurrent_kernel
&:$А2lstm_9/lstm_cell_9/bias
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
░
C	variables
rlayer_metrics
snon_trainable_variables
tmetrics
Dregularization_losses
Etrainable_variables
ulayer_regularization_losses

vlayers
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
░
S	variables
wlayer_metrics
xnon_trainable_variables
ymetrics
Tregularization_losses
Utrainable_variables
zlayer_regularization_losses

{layers
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
 "
trackable_list_wrapper
N
	|total
	}count
~	variables
	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
|0
}1"
trackable_list_wrapper
-
~	variables"
_generic_user_object
':%
АА2Adam/dense_8/kernel/m
 :А2Adam/dense_8/bias/m
&:$	А2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
1:/	А2 Adam/lstm_8/lstm_cell_8/kernel/m
;:9	@А2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
+:)А2Adam/lstm_8/lstm_cell_8/bias/m
1:/	@А2 Adam/lstm_9/lstm_cell_9/kernel/m
<::
АА2*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m
+:)А2Adam/lstm_9/lstm_cell_9/bias/m
':%
АА2Adam/dense_8/kernel/v
 :А2Adam/dense_8/bias/v
&:$	А2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
1:/	А2 Adam/lstm_8/lstm_cell_8/kernel/v
;:9	@А2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
+:)А2Adam/lstm_8/lstm_cell_8/bias/v
1:/	@А2 Adam/lstm_9/lstm_cell_9/kernel/v
<::
АА2*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v
+:)А2Adam/lstm_9/lstm_cell_9/bias/v
╤B╬
!__inference__wrapped_model_100225lstm_8_input"Ш
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
В2 
-__inference_sequential_2_layer_call_fn_101894
-__inference_sequential_2_layer_call_fn_102538
-__inference_sequential_2_layer_call_fn_102563
-__inference_sequential_2_layer_call_fn_102418└
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
ю2ы
H__inference_sequential_2_layer_call_and_return_conditional_losses_102885
H__inference_sequential_2_layer_call_and_return_conditional_losses_103221
H__inference_sequential_2_layer_call_and_return_conditional_losses_102449
H__inference_sequential_2_layer_call_and_return_conditional_losses_102480└
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
 2№
'__inference_lstm_8_layer_call_fn_103232
'__inference_lstm_8_layer_call_fn_103243
'__inference_lstm_8_layer_call_fn_103254
'__inference_lstm_8_layer_call_fn_103265╒
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
ы2ш
B__inference_lstm_8_layer_call_and_return_conditional_losses_103416
B__inference_lstm_8_layer_call_and_return_conditional_losses_103567
B__inference_lstm_8_layer_call_and_return_conditional_losses_103718
B__inference_lstm_8_layer_call_and_return_conditional_losses_103869╒
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
Т2П
*__inference_dropout_4_layer_call_fn_103874
*__inference_dropout_4_layer_call_fn_103879┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

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
╚2┼
E__inference_dropout_4_layer_call_and_return_conditional_losses_103884
E__inference_dropout_4_layer_call_and_return_conditional_losses_103896┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

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
 2№
'__inference_lstm_9_layer_call_fn_103907
'__inference_lstm_9_layer_call_fn_103918
'__inference_lstm_9_layer_call_fn_103929
'__inference_lstm_9_layer_call_fn_103940╒
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
ы2ш
B__inference_lstm_9_layer_call_and_return_conditional_losses_104091
B__inference_lstm_9_layer_call_and_return_conditional_losses_104242
B__inference_lstm_9_layer_call_and_return_conditional_losses_104393
B__inference_lstm_9_layer_call_and_return_conditional_losses_104544╒
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
Т2П
*__inference_dropout_5_layer_call_fn_104549
*__inference_dropout_5_layer_call_fn_104554┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

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
╚2┼
E__inference_dropout_5_layer_call_and_return_conditional_losses_104559
E__inference_dropout_5_layer_call_and_return_conditional_losses_104571┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

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
╥2╧
(__inference_dense_8_layer_call_fn_104580в
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
э2ъ
C__inference_dense_8_layer_call_and_return_conditional_losses_104591в
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
╥2╧
(__inference_dense_9_layer_call_fn_104600в
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
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_104610в
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
*__inference_reshape_4_layer_call_fn_104615в
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_104628в
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
╨B═
$__inference_signature_wrapper_102513lstm_8_input"Ф
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
а2Э
,__inference_lstm_cell_8_layer_call_fn_104645
,__inference_lstm_cell_8_layer_call_fn_104662╛
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
╓2╙
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104694
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104726╛
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
а2Э
,__inference_lstm_cell_9_layer_call_fn_104743
,__inference_lstm_cell_9_layer_call_fn_104760╛
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
╓2╙
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104792
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104824╛
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
 и
!__inference__wrapped_model_100225В
789:;<"#()9в6
/в,
*К'
lstm_8_input         
к "9к6
4
	reshape_4'К$
	reshape_4         е
C__inference_dense_8_layer_call_and_return_conditional_losses_104591^"#0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ }
(__inference_dense_8_layer_call_fn_104580Q"#0в-
&в#
!К
inputs         А
к "К         Ад
C__inference_dense_9_layer_call_and_return_conditional_losses_104610]()0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ |
(__inference_dense_9_layer_call_fn_104600P()0в-
&в#
!К
inputs         А
к "К         н
E__inference_dropout_4_layer_call_and_return_conditional_losses_103884d7в4
-в*
$К!
inputs         @
p 
к ")в&
К
0         @
Ъ н
E__inference_dropout_4_layer_call_and_return_conditional_losses_103896d7в4
-в*
$К!
inputs         @
p
к ")в&
К
0         @
Ъ Е
*__inference_dropout_4_layer_call_fn_103874W7в4
-в*
$К!
inputs         @
p 
к "К         @Е
*__inference_dropout_4_layer_call_fn_103879W7в4
-в*
$К!
inputs         @
p
к "К         @з
E__inference_dropout_5_layer_call_and_return_conditional_losses_104559^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ з
E__inference_dropout_5_layer_call_and_return_conditional_losses_104571^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ 
*__inference_dropout_5_layer_call_fn_104549Q4в1
*в'
!К
inputs         А
p 
к "К         А
*__inference_dropout_5_layer_call_fn_104554Q4в1
*в'
!К
inputs         А
p
к "К         А╤
B__inference_lstm_8_layer_call_and_return_conditional_losses_103416К789OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "2в/
(К%
0                  @
Ъ ╤
B__inference_lstm_8_layer_call_and_return_conditional_losses_103567К789OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "2в/
(К%
0                  @
Ъ ╖
B__inference_lstm_8_layer_call_and_return_conditional_losses_103718q789?в<
5в2
$К!
inputs         

 
p 

 
к ")в&
К
0         @
Ъ ╖
B__inference_lstm_8_layer_call_and_return_conditional_losses_103869q789?в<
5в2
$К!
inputs         

 
p

 
к ")в&
К
0         @
Ъ и
'__inference_lstm_8_layer_call_fn_103232}789OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%К"                  @и
'__inference_lstm_8_layer_call_fn_103243}789OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%К"                  @П
'__inference_lstm_8_layer_call_fn_103254d789?в<
5в2
$К!
inputs         

 
p 

 
к "К         @П
'__inference_lstm_8_layer_call_fn_103265d789?в<
5в2
$К!
inputs         

 
p

 
к "К         @─
B__inference_lstm_9_layer_call_and_return_conditional_losses_104091~:;<OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "&в#
К
0         А
Ъ ─
B__inference_lstm_9_layer_call_and_return_conditional_losses_104242~:;<OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "&в#
К
0         А
Ъ ┤
B__inference_lstm_9_layer_call_and_return_conditional_losses_104393n:;<?в<
5в2
$К!
inputs         @

 
p 

 
к "&в#
К
0         А
Ъ ┤
B__inference_lstm_9_layer_call_and_return_conditional_losses_104544n:;<?в<
5в2
$К!
inputs         @

 
p

 
к "&в#
К
0         А
Ъ Ь
'__inference_lstm_9_layer_call_fn_103907q:;<OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "К         АЬ
'__inference_lstm_9_layer_call_fn_103918q:;<OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "К         АМ
'__inference_lstm_9_layer_call_fn_103929a:;<?в<
5в2
$К!
inputs         @

 
p 

 
к "К         АМ
'__inference_lstm_9_layer_call_fn_103940a:;<?в<
5в2
$К!
inputs         @

 
p

 
к "К         А╔
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104694¤789Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p 
к "sвp
iвf
К
0/0         @
EЪB
К
0/1/0         @
К
0/1/1         @
Ъ ╔
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_104726¤789Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p
к "sвp
iвf
К
0/0         @
EЪB
К
0/1/0         @
К
0/1/1         @
Ъ Ю
,__inference_lstm_cell_8_layer_call_fn_104645э789Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p 
к "cв`
К
0         @
AЪ>
К
1/0         @
К
1/1         @Ю
,__inference_lstm_cell_8_layer_call_fn_104662э789Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p
к "cв`
К
0         @
AЪ>
К
1/0         @
К
1/1         @╬
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104792В:;<Вв
xвu
 К
inputs         @
MвJ
#К 
states/0         А
#К 
states/1         А
p 
к "vвs
lвi
К
0/0         А
GЪD
 К
0/1/0         А
 К
0/1/1         А
Ъ ╬
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_104824В:;<Вв
xвu
 К
inputs         @
MвJ
#К 
states/0         А
#К 
states/1         А
p
к "vвs
lвi
К
0/0         А
GЪD
 К
0/1/0         А
 К
0/1/1         А
Ъ г
,__inference_lstm_cell_9_layer_call_fn_104743Є:;<Вв
xвu
 К
inputs         @
MвJ
#К 
states/0         А
#К 
states/1         А
p 
к "fвc
К
0         А
CЪ@
К
1/0         А
К
1/1         Аг
,__inference_lstm_cell_9_layer_call_fn_104760Є:;<Вв
xвu
 К
inputs         @
MвJ
#К 
states/0         А
#К 
states/1         А
p
к "fвc
К
0         А
CЪ@
К
1/0         А
К
1/1         Ае
E__inference_reshape_4_layer_call_and_return_conditional_losses_104628\/в,
%в"
 К
inputs         
к ")в&
К
0         
Ъ }
*__inference_reshape_4_layer_call_fn_104615O/в,
%в"
 К
inputs         
к "К         ╞
H__inference_sequential_2_layer_call_and_return_conditional_losses_102449z
789:;<"#()Aв>
7в4
*К'
lstm_8_input         
p 

 
к ")в&
К
0         
Ъ ╞
H__inference_sequential_2_layer_call_and_return_conditional_losses_102480z
789:;<"#()Aв>
7в4
*К'
lstm_8_input         
p

 
к ")в&
К
0         
Ъ └
H__inference_sequential_2_layer_call_and_return_conditional_losses_102885t
789:;<"#();в8
1в.
$К!
inputs         
p 

 
к ")в&
К
0         
Ъ └
H__inference_sequential_2_layer_call_and_return_conditional_losses_103221t
789:;<"#();в8
1в.
$К!
inputs         
p

 
к ")в&
К
0         
Ъ Ю
-__inference_sequential_2_layer_call_fn_101894m
789:;<"#()Aв>
7в4
*К'
lstm_8_input         
p 

 
к "К         Ю
-__inference_sequential_2_layer_call_fn_102418m
789:;<"#()Aв>
7в4
*К'
lstm_8_input         
p

 
к "К         Ш
-__inference_sequential_2_layer_call_fn_102538g
789:;<"#();в8
1в.
$К!
inputs         
p 

 
к "К         Ш
-__inference_sequential_2_layer_call_fn_102563g
789:;<"#();в8
1в.
$К!
inputs         
p

 
к "К         ╗
$__inference_signature_wrapper_102513Т
789:;<"#()IвF
в 
?к<
:
lstm_8_input*К'
lstm_8_input         "9к6
4
	reshape_4'К$
	reshape_4         