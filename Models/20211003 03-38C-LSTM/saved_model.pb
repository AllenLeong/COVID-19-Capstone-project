Ол,
Чш
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
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8╢И*
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
: *
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
: *
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
: @*
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes
:@*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:		*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:	*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:	*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
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
Т
lstm_34/lstm_cell_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*,
shared_namelstm_34/lstm_cell_34/kernel
Л
/lstm_34/lstm_cell_34/kernel/Read/ReadVariableOpReadVariableOplstm_34/lstm_cell_34/kernel*
_output_shapes

:@$*
dtype0
ж
%lstm_34/lstm_cell_34/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*6
shared_name'%lstm_34/lstm_cell_34/recurrent_kernel
Я
9lstm_34/lstm_cell_34/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_34/lstm_cell_34/recurrent_kernel*
_output_shapes

:	$*
dtype0
К
lstm_34/lstm_cell_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_namelstm_34/lstm_cell_34/bias
Г
-lstm_34/lstm_cell_34/bias/Read/ReadVariableOpReadVariableOplstm_34/lstm_cell_34/bias*
_output_shapes
:$*
dtype0
Т
lstm_35/lstm_cell_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*,
shared_namelstm_35/lstm_cell_35/kernel
Л
/lstm_35/lstm_cell_35/kernel/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell_35/kernel*
_output_shapes

:	$*
dtype0
ж
%lstm_35/lstm_cell_35/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*6
shared_name'%lstm_35/lstm_cell_35/recurrent_kernel
Я
9lstm_35/lstm_cell_35/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_35/lstm_cell_35/recurrent_kernel*
_output_shapes

:	$*
dtype0
К
lstm_35/lstm_cell_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_namelstm_35/lstm_cell_35/bias
Г
-lstm_35/lstm_cell_35/bias/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell_35/bias*
_output_shapes
:$*
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
М
Adam/conv1d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_6/kernel/m
Е
*Adam/conv1d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/kernel/m*"
_output_shapes
: *
dtype0
А
Adam/conv1d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_6/bias/m
y
(Adam/conv1d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/bias/m*
_output_shapes
: *
dtype0
М
Adam/conv1d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv1d_7/kernel/m
Е
*Adam/conv1d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/kernel/m*"
_output_shapes
: @*
dtype0
А
Adam/conv1d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_7/bias/m
y
(Adam/conv1d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/bias/m*
_output_shapes
:@*
dtype0
И
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*'
shared_nameAdam/dense_34/kernel/m
Б
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes

:		*
dtype0
А
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
:	*
dtype0
И
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*'
shared_nameAdam/dense_35/kernel/m
Б
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:	*
dtype0
А
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0
а
"Adam/lstm_34/lstm_cell_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*3
shared_name$"Adam/lstm_34/lstm_cell_34/kernel/m
Щ
6Adam/lstm_34/lstm_cell_34/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_34/lstm_cell_34/kernel/m*
_output_shapes

:@$*
dtype0
┤
,Adam/lstm_34/lstm_cell_34/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*=
shared_name.,Adam/lstm_34/lstm_cell_34/recurrent_kernel/m
н
@Adam/lstm_34/lstm_cell_34/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_34/lstm_cell_34/recurrent_kernel/m*
_output_shapes

:	$*
dtype0
Ш
 Adam/lstm_34/lstm_cell_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*1
shared_name" Adam/lstm_34/lstm_cell_34/bias/m
С
4Adam/lstm_34/lstm_cell_34/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_34/lstm_cell_34/bias/m*
_output_shapes
:$*
dtype0
а
"Adam/lstm_35/lstm_cell_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*3
shared_name$"Adam/lstm_35/lstm_cell_35/kernel/m
Щ
6Adam/lstm_35/lstm_cell_35/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_35/lstm_cell_35/kernel/m*
_output_shapes

:	$*
dtype0
┤
,Adam/lstm_35/lstm_cell_35/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*=
shared_name.,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m
н
@Adam/lstm_35/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m*
_output_shapes

:	$*
dtype0
Ш
 Adam/lstm_35/lstm_cell_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*1
shared_name" Adam/lstm_35/lstm_cell_35/bias/m
С
4Adam/lstm_35/lstm_cell_35/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_35/lstm_cell_35/bias/m*
_output_shapes
:$*
dtype0
М
Adam/conv1d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_6/kernel/v
Е
*Adam/conv1d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/kernel/v*"
_output_shapes
: *
dtype0
А
Adam/conv1d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_6/bias/v
y
(Adam/conv1d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/bias/v*
_output_shapes
: *
dtype0
М
Adam/conv1d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv1d_7/kernel/v
Е
*Adam/conv1d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/kernel/v*"
_output_shapes
: @*
dtype0
А
Adam/conv1d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_7/bias/v
y
(Adam/conv1d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/bias/v*
_output_shapes
:@*
dtype0
И
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*'
shared_nameAdam/dense_34/kernel/v
Б
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes

:		*
dtype0
А
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
:	*
dtype0
И
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*'
shared_nameAdam/dense_35/kernel/v
Б
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:	*
dtype0
А
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0
а
"Adam/lstm_34/lstm_cell_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*3
shared_name$"Adam/lstm_34/lstm_cell_34/kernel/v
Щ
6Adam/lstm_34/lstm_cell_34/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_34/lstm_cell_34/kernel/v*
_output_shapes

:@$*
dtype0
┤
,Adam/lstm_34/lstm_cell_34/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*=
shared_name.,Adam/lstm_34/lstm_cell_34/recurrent_kernel/v
н
@Adam/lstm_34/lstm_cell_34/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_34/lstm_cell_34/recurrent_kernel/v*
_output_shapes

:	$*
dtype0
Ш
 Adam/lstm_34/lstm_cell_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*1
shared_name" Adam/lstm_34/lstm_cell_34/bias/v
С
4Adam/lstm_34/lstm_cell_34/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_34/lstm_cell_34/bias/v*
_output_shapes
:$*
dtype0
а
"Adam/lstm_35/lstm_cell_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*3
shared_name$"Adam/lstm_35/lstm_cell_35/kernel/v
Щ
6Adam/lstm_35/lstm_cell_35/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_35/lstm_cell_35/kernel/v*
_output_shapes

:	$*
dtype0
┤
,Adam/lstm_35/lstm_cell_35/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*=
shared_name.,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v
н
@Adam/lstm_35/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v*
_output_shapes

:	$*
dtype0
Ш
 Adam/lstm_35/lstm_cell_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*1
shared_name" Adam/lstm_35/lstm_cell_35/bias/v
С
4Adam/lstm_35/lstm_cell_35/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_35/lstm_cell_35/bias/v*
_output_shapes
:$*
dtype0

NoOpNoOp
зU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*тT
value╪TB╒T B╬T
П
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
 	keras_api
l
!cell
"
state_spec
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
l
+cell
,
state_spec
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
╪
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratemвmгmдmе5mж6mз;mи<mйJmкKmлLmмMmнNmоOmпv░v▒v▓v│5v┤6v╡;v╢<v╖Jv╕Kv╣Lv║Mv╗Nv╝Ov╜
 
f
0
1
2
3
J4
K5
L6
M7
N8
O9
510
611
;12
<13
f
0
1
2
3
J4
K5
L6
M7
N8
O9
510
611
;12
<13
н
Player_regularization_losses
regularization_losses
Qlayer_metrics
trainable_variables

Rlayers
Snon_trainable_variables
Tmetrics
	variables
 
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Ulayer_regularization_losses
regularization_losses
Vlayer_metrics
trainable_variables

Wlayers
Xnon_trainable_variables
Ymetrics
	variables
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Zlayer_regularization_losses
regularization_losses
[layer_metrics
trainable_variables

\layers
]non_trainable_variables
^metrics
	variables
 
 
 
н
_layer_regularization_losses
regularization_losses
`layer_metrics
trainable_variables

alayers
bnon_trainable_variables
cmetrics
	variables
О
d
state_size

Jkernel
Krecurrent_kernel
Lbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
 
 

J0
K1
L2

J0
K1
L2
╣
ilayer_regularization_losses
#regularization_losses
jlayer_metrics
$trainable_variables

klayers

lstates
mnon_trainable_variables
nmetrics
%	variables
 
 
 
н
olayer_regularization_losses
'regularization_losses
player_metrics
(trainable_variables

qlayers
rnon_trainable_variables
smetrics
)	variables
О
t
state_size

Mkernel
Nrecurrent_kernel
Obias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
 
 

M0
N1
O2

M0
N1
O2
╣
ylayer_regularization_losses
-regularization_losses
zlayer_metrics
.trainable_variables

{layers

|states
}non_trainable_variables
~metrics
/	variables
 
 
 
▒
layer_regularization_losses
1regularization_losses
Аlayer_metrics
2trainable_variables
Бlayers
Вnon_trainable_variables
Гmetrics
3	variables
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
▓
 Дlayer_regularization_losses
7regularization_losses
Еlayer_metrics
8trainable_variables
Жlayers
Зnon_trainable_variables
Иmetrics
9	variables
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
▓
 Йlayer_regularization_losses
=regularization_losses
Кlayer_metrics
>trainable_variables
Лlayers
Мnon_trainable_variables
Нmetrics
?	variables
 
 
 
▓
 Оlayer_regularization_losses
Aregularization_losses
Пlayer_metrics
Btrainable_variables
Рlayers
Сnon_trainable_variables
Тmetrics
C	variables
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
VARIABLE_VALUElstm_34/lstm_cell_34/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_34/lstm_cell_34/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_34/lstm_cell_34/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_35/lstm_cell_35/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_35/lstm_cell_35/recurrent_kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_35/lstm_cell_35/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
 
F
0
1
2
3
4
5
6
7
	8

9
 

У0
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

J0
K1
L2

J0
K1
L2
▓
 Фlayer_regularization_losses
eregularization_losses
Хlayer_metrics
ftrainable_variables
Цlayers
Чnon_trainable_variables
Шmetrics
g	variables
 
 

!0
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
M0
N1
O2

M0
N1
O2
▓
 Щlayer_regularization_losses
uregularization_losses
Ъlayer_metrics
vtrainable_variables
Ыlayers
Ьnon_trainable_variables
Эmetrics
w	variables
 
 

+0
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
8

Юtotal

Яcount
а	variables
б	keras_api
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

Ю0
Я1

а	variables
~|
VARIABLE_VALUEAdam/conv1d_6/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_6/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_7/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_7/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_34/lstm_cell_34/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_34/lstm_cell_34/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_34/lstm_cell_34/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_35/lstm_cell_35/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_35/lstm_cell_35/recurrent_kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_35/lstm_cell_35/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_6/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_6/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_7/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_7/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_34/lstm_cell_34/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_34/lstm_cell_34/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_34/lstm_cell_34/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_35/lstm_cell_35/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_35/lstm_cell_35/recurrent_kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_35/lstm_cell_35/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Й
serving_default_conv1d_6_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_6_inputconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biaslstm_34/lstm_cell_34/kernel%lstm_34/lstm_cell_34/recurrent_kernellstm_34/lstm_cell_34/biaslstm_35/lstm_cell_35/kernel%lstm_35/lstm_cell_35/recurrent_kernellstm_35/lstm_cell_35/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_321669
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_34/lstm_cell_34/kernel/Read/ReadVariableOp9lstm_34/lstm_cell_34/recurrent_kernel/Read/ReadVariableOp-lstm_34/lstm_cell_34/bias/Read/ReadVariableOp/lstm_35/lstm_cell_35/kernel/Read/ReadVariableOp9lstm_35/lstm_cell_35/recurrent_kernel/Read/ReadVariableOp-lstm_35/lstm_cell_35/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv1d_6/kernel/m/Read/ReadVariableOp(Adam/conv1d_6/bias/m/Read/ReadVariableOp*Adam/conv1d_7/kernel/m/Read/ReadVariableOp(Adam/conv1d_7/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp6Adam/lstm_34/lstm_cell_34/kernel/m/Read/ReadVariableOp@Adam/lstm_34/lstm_cell_34/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_34/lstm_cell_34/bias/m/Read/ReadVariableOp6Adam/lstm_35/lstm_cell_35/kernel/m/Read/ReadVariableOp@Adam/lstm_35/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_35/lstm_cell_35/bias/m/Read/ReadVariableOp*Adam/conv1d_6/kernel/v/Read/ReadVariableOp(Adam/conv1d_6/bias/v/Read/ReadVariableOp*Adam/conv1d_7/kernel/v/Read/ReadVariableOp(Adam/conv1d_7/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp6Adam/lstm_34/lstm_cell_34/kernel/v/Read/ReadVariableOp@Adam/lstm_34/lstm_cell_34/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_34/lstm_cell_34/bias/v/Read/ReadVariableOp6Adam/lstm_35/lstm_cell_35/kernel/v/Read/ReadVariableOp@Adam/lstm_35/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_35/lstm_cell_35/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
__inference__traced_save_324298
Ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_34/lstm_cell_34/kernel%lstm_34/lstm_cell_34/recurrent_kernellstm_34/lstm_cell_34/biaslstm_35/lstm_cell_35/kernel%lstm_35/lstm_cell_35/recurrent_kernellstm_35/lstm_cell_35/biastotalcountAdam/conv1d_6/kernel/mAdam/conv1d_6/bias/mAdam/conv1d_7/kernel/mAdam/conv1d_7/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/m"Adam/lstm_34/lstm_cell_34/kernel/m,Adam/lstm_34/lstm_cell_34/recurrent_kernel/m Adam/lstm_34/lstm_cell_34/bias/m"Adam/lstm_35/lstm_cell_35/kernel/m,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m Adam/lstm_35/lstm_cell_35/bias/mAdam/conv1d_6/kernel/vAdam/conv1d_6/bias/vAdam/conv1d_7/kernel/vAdam/conv1d_7/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v"Adam/lstm_34/lstm_cell_34/kernel/v,Adam/lstm_34/lstm_cell_34/recurrent_kernel/v Adam/lstm_34/lstm_cell_34/bias/v"Adam/lstm_35/lstm_cell_35/kernel/v,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v Adam/lstm_35/lstm_cell_35/bias/v*=
Tin6
422*
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
"__inference__traced_restore_324455ад(
╒
├
while_cond_323461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323461___redundant_placeholder04
0while_while_cond_323461___redundant_placeholder14
0while_while_cond_323461___redundant_placeholder24
0while_while_cond_323461___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
В
ї
D__inference_dense_34_layer_call_and_return_conditional_losses_323895

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╒
├
while_cond_319528
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_319528___redundant_placeholder04
0while_while_cond_319528___redundant_placeholder14
0while_while_cond_319528___redundant_placeholder24
0while_while_cond_319528___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
ж

ї
D__inference_dense_35_layer_call_and_return_conditional_losses_323914

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
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
_construction_contextkEagerRuntime**
_input_shapes
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
г
L
0__inference_max_pooling1d_2_layer_call_fn_322504

inputs
identity▀
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3192142
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
┐^
М
'sequential_12_lstm_35_while_body_319095H
Dsequential_12_lstm_35_while_sequential_12_lstm_35_while_loop_counterN
Jsequential_12_lstm_35_while_sequential_12_lstm_35_while_maximum_iterations+
'sequential_12_lstm_35_while_placeholder-
)sequential_12_lstm_35_while_placeholder_1-
)sequential_12_lstm_35_while_placeholder_2-
)sequential_12_lstm_35_while_placeholder_3G
Csequential_12_lstm_35_while_sequential_12_lstm_35_strided_slice_1_0Г
sequential_12_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_35_tensorarrayunstack_tensorlistfromtensor_0[
Isequential_12_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	$]
Ksequential_12_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$X
Jsequential_12_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:$(
$sequential_12_lstm_35_while_identity*
&sequential_12_lstm_35_while_identity_1*
&sequential_12_lstm_35_while_identity_2*
&sequential_12_lstm_35_while_identity_3*
&sequential_12_lstm_35_while_identity_4*
&sequential_12_lstm_35_while_identity_5E
Asequential_12_lstm_35_while_sequential_12_lstm_35_strided_slice_1Б
}sequential_12_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_35_tensorarrayunstack_tensorlistfromtensorY
Gsequential_12_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	$[
Isequential_12_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	$V
Hsequential_12_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:$Ив?sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpв>sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpв@sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpя
Msequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2O
Msequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_35_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_35_while_placeholderVsequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02A
?sequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItemК
>sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02@
>sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpо
/sequential_12/lstm_35/while/lstm_cell_35/MatMulMatMulFsequential_12/lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $21
/sequential_12/lstm_35/while/lstm_cell_35/MatMulР
@sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02B
@sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpЧ
1sequential_12/lstm_35/while/lstm_cell_35/MatMul_1MatMul)sequential_12_lstm_35_while_placeholder_2Hsequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $23
1sequential_12/lstm_35/while/lstm_cell_35/MatMul_1П
,sequential_12/lstm_35/while/lstm_cell_35/addAddV29sequential_12/lstm_35/while/lstm_cell_35/MatMul:product:0;sequential_12/lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2.
,sequential_12/lstm_35/while/lstm_cell_35/addЙ
?sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02A
?sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpЬ
0sequential_12/lstm_35/while/lstm_cell_35/BiasAddBiasAdd0sequential_12/lstm_35/while/lstm_cell_35/add:z:0Gsequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $22
0sequential_12/lstm_35/while/lstm_cell_35/BiasAdd╢
8sequential_12/lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_35/while/lstm_cell_35/split/split_dimу
.sequential_12/lstm_35/while/lstm_cell_35/splitSplitAsequential_12/lstm_35/while/lstm_cell_35/split/split_dim:output:09sequential_12/lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split20
.sequential_12/lstm_35/while/lstm_cell_35/split┌
0sequential_12/lstm_35/while/lstm_cell_35/SigmoidSigmoid7sequential_12/lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	22
0sequential_12/lstm_35/while/lstm_cell_35/Sigmoid▐
2sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid7sequential_12/lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	24
2sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_1°
,sequential_12/lstm_35/while/lstm_cell_35/mulMul6sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_1:y:0)sequential_12_lstm_35_while_placeholder_3*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_35/while/lstm_cell_35/mul╤
-sequential_12/lstm_35/while/lstm_cell_35/ReluRelu7sequential_12/lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2/
-sequential_12/lstm_35/while/lstm_cell_35/ReluМ
.sequential_12/lstm_35/while/lstm_cell_35/mul_1Mul4sequential_12/lstm_35/while/lstm_cell_35/Sigmoid:y:0;sequential_12/lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_35/while/lstm_cell_35/mul_1Б
.sequential_12/lstm_35/while/lstm_cell_35/add_1AddV20sequential_12/lstm_35/while/lstm_cell_35/mul:z:02sequential_12/lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_35/while/lstm_cell_35/add_1▐
2sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid7sequential_12/lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	24
2sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_2╨
/sequential_12/lstm_35/while/lstm_cell_35/Relu_1Relu2sequential_12/lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	21
/sequential_12/lstm_35/while/lstm_cell_35/Relu_1Р
.sequential_12/lstm_35/while/lstm_cell_35/mul_2Mul6sequential_12/lstm_35/while/lstm_cell_35/Sigmoid_2:y:0=sequential_12/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_35/while/lstm_cell_35/mul_2╬
@sequential_12/lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_35_while_placeholder_1'sequential_12_lstm_35_while_placeholder2sequential_12/lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_35/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_12/lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_35/while/add/y┴
sequential_12/lstm_35/while/addAddV2'sequential_12_lstm_35_while_placeholder*sequential_12/lstm_35/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_35/while/addМ
#sequential_12/lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_35/while/add_1/yф
!sequential_12/lstm_35/while/add_1AddV2Dsequential_12_lstm_35_while_sequential_12_lstm_35_while_loop_counter,sequential_12/lstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_35/while/add_1├
$sequential_12/lstm_35/while/IdentityIdentity%sequential_12/lstm_35/while/add_1:z:0!^sequential_12/lstm_35/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_35/while/Identityь
&sequential_12/lstm_35/while/Identity_1IdentityJsequential_12_lstm_35_while_sequential_12_lstm_35_while_maximum_iterations!^sequential_12/lstm_35/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_35/while/Identity_1┼
&sequential_12/lstm_35/while/Identity_2Identity#sequential_12/lstm_35/while/add:z:0!^sequential_12/lstm_35/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_35/while/Identity_2Є
&sequential_12/lstm_35/while/Identity_3IdentityPsequential_12/lstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_35/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_35/while/Identity_3х
&sequential_12/lstm_35/while/Identity_4Identity2sequential_12/lstm_35/while/lstm_cell_35/mul_2:z:0!^sequential_12/lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_35/while/Identity_4х
&sequential_12/lstm_35/while/Identity_5Identity2sequential_12/lstm_35/while/lstm_cell_35/add_1:z:0!^sequential_12/lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_35/while/Identity_5╠
 sequential_12/lstm_35/while/NoOpNoOp@^sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?^sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpA^sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_35/while/NoOp"U
$sequential_12_lstm_35_while_identity-sequential_12/lstm_35/while/Identity:output:0"Y
&sequential_12_lstm_35_while_identity_1/sequential_12/lstm_35/while/Identity_1:output:0"Y
&sequential_12_lstm_35_while_identity_2/sequential_12/lstm_35/while/Identity_2:output:0"Y
&sequential_12_lstm_35_while_identity_3/sequential_12/lstm_35/while/Identity_3:output:0"Y
&sequential_12_lstm_35_while_identity_4/sequential_12/lstm_35/while/Identity_4:output:0"Y
&sequential_12_lstm_35_while_identity_5/sequential_12/lstm_35/while/Identity_5:output:0"Ц
Hsequential_12_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resourceJsequential_12_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"Ш
Isequential_12_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resourceKsequential_12_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"Ф
Gsequential_12_lstm_35_while_lstm_cell_35_matmul_readvariableop_resourceIsequential_12_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"И
Asequential_12_lstm_35_while_sequential_12_lstm_35_strided_slice_1Csequential_12_lstm_35_while_sequential_12_lstm_35_strided_slice_1_0"А
}sequential_12_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_35_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2В
?sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?sequential_12/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2А
>sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp>sequential_12/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2Д
@sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp@sequential_12/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
щ
Б
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_320081

inputs

states
states_10
matmul_readvariableop_resource:	$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_namestates:OK
'
_output_shapes
:         	
 
_user_specified_namestates
╥%
▌
while_body_320159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_35_320183_0:	$-
while_lstm_cell_35_320185_0:	$)
while_lstm_cell_35_320187_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_35_320183:	$+
while_lstm_cell_35_320185:	$'
while_lstm_cell_35_320187:$Ив*while/lstm_cell_35/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_320183_0while_lstm_cell_35_320185_0while_lstm_cell_35_320187_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3200812,
*while/lstm_cell_35/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
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
while_lstm_cell_35_320183while_lstm_cell_35_320183_0"8
while_lstm_cell_35_320185while_lstm_cell_35_320185_0"8
while_lstm_cell_35_320187while_lstm_cell_35_320187_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
Л?
╩
while_body_323613
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
│
є
-__inference_lstm_cell_35_layer_call_fn_324064

inputs
states_0
states_1
unknown:	$
	unknown_0:	$
	unknown_1:$
identity

identity_1

identity_2ИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3200812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         	2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
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
л
У
D__inference_conv1d_7_layer_call_and_return_conditional_losses_322499

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╢
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Л?
╩
while_body_323462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
ШF
В
C__inference_lstm_35_layer_call_and_return_conditional_losses_320228

inputs%
lstm_cell_35_320146:	$%
lstm_cell_35_320148:	$!
lstm_cell_35_320150:$
identityИв$lstm_cell_35/StatefulPartitionedCallвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
 :                  	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2Э
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_320146lstm_cell_35_320148lstm_cell_35_320150*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3200812&
$lstm_cell_35/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counter└
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_320146lstm_cell_35_320148lstm_cell_35_320150*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320159*
condR
while_cond_320158*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity}
NoOpNoOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
д
g
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322525

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:         @2

ExpandDimsЮ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
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
С
g
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322517

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╒
├
while_cond_322937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322937___redundant_placeholder04
0while_while_cond_322937___redundant_placeholder14
0while_while_cond_322937___redundant_placeholder24
0while_while_cond_322937___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Ц\
Ъ
C__inference_lstm_34_layer_call_and_return_conditional_losses_322720
inputs_0=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileF
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322636*
condR
while_cond_322635*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
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
 :                  	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
═
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_321202

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
:         	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         	*
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
:         	2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         	2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Л?
╩
while_body_323089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
ё
Ц
)__inference_dense_34_layer_call_fn_323884

inputs
unknown:		
	unknown_0:	
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3208912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
═
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_323200

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
:         	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         	*
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
:         	2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         	2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
л
У
D__inference_conv1d_7_layer_call_and_return_conditional_losses_320535

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╢
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
╫[
Ш
C__inference_lstm_34_layer_call_and_return_conditional_losses_320700

inputs=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320616*
condR
while_cond_320615*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
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
:         	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╥%
▌
while_body_319949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_35_319973_0:	$-
while_lstm_cell_35_319975_0:	$)
while_lstm_cell_35_319977_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_35_319973:	$+
while_lstm_cell_35_319975:	$'
while_lstm_cell_35_319977:$Ив*while/lstm_cell_35/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_319973_0while_lstm_cell_35_319975_0while_lstm_cell_35_319977_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3199352,
*while/lstm_cell_35/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
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
while_lstm_cell_35_319973while_lstm_cell_35_319973_0"8
while_lstm_cell_35_319975while_lstm_cell_35_319975_0"8
while_lstm_cell_35_319977while_lstm_cell_35_319977_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_319948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_319948___redundant_placeholder04
0while_while_cond_319948___redundant_placeholder14
0while_while_cond_319948___redundant_placeholder24
0while_while_cond_319948___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Х
ю
.__inference_sequential_12_layer_call_fn_321735

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@$
	unknown_4:	$
	unknown_5:$
	unknown_6:	$
	unknown_7:	$
	unknown_8:$
	unknown_9:		

unknown_10:	

unknown_11:	

unknown_12:
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3214802
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╫[
Ш
C__inference_lstm_34_layer_call_and_return_conditional_losses_323173

inputs=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323089*
condR
while_cond_323088*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
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
:         	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Й
b
F__inference_reshape_17_layer_call_and_return_conditional_losses_320926

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
while_cond_323088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323088___redundant_placeholder04
0while_while_cond_323088___redundant_placeholder14
0while_while_cond_323088___redundant_placeholder24
0while_while_cond_323088___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
м
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_321006

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         	2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         	2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
Л?
╩
while_body_320616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_321088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_321088___redundant_placeholder04
0while_while_cond_321088___redundant_placeholder14
0while_while_cond_321088___redundant_placeholder24
0while_while_cond_321088___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
щ
Б
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_319935

inputs

states
states_10
matmul_readvariableop_resource:	$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_namestates:OK
'
_output_shapes
:         	
 
_user_specified_namestates
ё
Г
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_324030

inputs
states_0
states_10
matmul_readvariableop_resource:@$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
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
Й
b
F__inference_reshape_17_layer_call_and_return_conditional_losses_323932

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
╛/
Ф
I__inference_sequential_12_layer_call_and_return_conditional_losses_321586
conv1d_6_input%
conv1d_6_321547: 
conv1d_6_321549: %
conv1d_7_321552: @
conv1d_7_321554:@ 
lstm_34_321558:@$ 
lstm_34_321560:	$
lstm_34_321562:$ 
lstm_35_321566:	$ 
lstm_35_321568:	$
lstm_35_321570:$!
dense_34_321574:		
dense_34_321576:	!
dense_35_321579:	
dense_35_321581:
identityИв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallвlstm_34/StatefulPartitionedCallвlstm_35/StatefulPartitionedCallа
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallconv1d_6_inputconv1d_6_321547conv1d_6_321549*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3205132"
 conv1d_6/StatefulPartitionedCall╗
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0conv1d_7_321552conv1d_7_321554*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3205352"
 conv1d_7/StatefulPartitionedCallР
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3205482!
max_pooling1d_2/PartitionedCall╟
lstm_34/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0lstm_34_321558lstm_34_321560lstm_34_321562*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3207002!
lstm_34/StatefulPartitionedCallА
dropout_18/PartitionedCallPartitionedCall(lstm_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3207132
dropout_18/PartitionedCall╛
lstm_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_35_321566lstm_35_321568lstm_35_321570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3208652!
lstm_35/StatefulPartitionedCall№
dropout_19/PartitionedCallPartitionedCall(lstm_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3208782
dropout_19/PartitionedCall▒
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_34_321574dense_34_321576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3208912"
 dense_34/StatefulPartitionedCall╖
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_321579dense_35_321581*
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
GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_3209072"
 dense_35/StatefulPartitionedCallБ
reshape_17/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *O
fJRH
F__inference_reshape_17_layer_call_and_return_conditional_losses_3209262
reshape_17/PartitionedCallВ
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЮ
NoOpNoOp!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^lstm_34/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
lstm_34/StatefulPartitionedCalllstm_34/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
д
g
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_320548

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:         @2

ExpandDimsЮ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
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
╒
├
while_cond_320615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_320615___redundant_placeholder04
0while_while_cond_320615___redundant_placeholder14
0while_while_cond_320615___redundant_placeholder24
0while_while_cond_320615___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
└J
╩

lstm_34_while_body_321830,
(lstm_34_while_lstm_34_while_loop_counter2
.lstm_34_while_lstm_34_while_maximum_iterations
lstm_34_while_placeholder
lstm_34_while_placeholder_1
lstm_34_while_placeholder_2
lstm_34_while_placeholder_3+
'lstm_34_while_lstm_34_strided_slice_1_0g
clstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0:@$O
=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$J
<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0:$
lstm_34_while_identity
lstm_34_while_identity_1
lstm_34_while_identity_2
lstm_34_while_identity_3
lstm_34_while_identity_4
lstm_34_while_identity_5)
%lstm_34_while_lstm_34_strided_slice_1e
alstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensorK
9lstm_34_while_lstm_cell_34_matmul_readvariableop_resource:@$M
;lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource:	$H
:lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource:$Ив1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpв0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpв2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp╙
?lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2A
?lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_34/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0lstm_34_while_placeholderHlstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype023
1lstm_34/while/TensorArrayV2Read/TensorListGetItemр
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype022
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpЎ
!lstm_34/while/lstm_cell_34/MatMulMatMul8lstm_34/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2#
!lstm_34/while/lstm_cell_34/MatMulц
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype024
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp▀
#lstm_34/while/lstm_cell_34/MatMul_1MatMullstm_34_while_placeholder_2:lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2%
#lstm_34/while/lstm_cell_34/MatMul_1╫
lstm_34/while/lstm_cell_34/addAddV2+lstm_34/while/lstm_cell_34/MatMul:product:0-lstm_34/while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2 
lstm_34/while/lstm_cell_34/add▀
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype023
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpф
"lstm_34/while/lstm_cell_34/BiasAddBiasAdd"lstm_34/while/lstm_cell_34/add:z:09lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2$
"lstm_34/while/lstm_cell_34/BiasAddЪ
*lstm_34/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_34/while/lstm_cell_34/split/split_dimл
 lstm_34/while/lstm_cell_34/splitSplit3lstm_34/while/lstm_cell_34/split/split_dim:output:0+lstm_34/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2"
 lstm_34/while/lstm_cell_34/split░
"lstm_34/while/lstm_cell_34/SigmoidSigmoid)lstm_34/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2$
"lstm_34/while/lstm_cell_34/Sigmoid┤
$lstm_34/while/lstm_cell_34/Sigmoid_1Sigmoid)lstm_34/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2&
$lstm_34/while/lstm_cell_34/Sigmoid_1└
lstm_34/while/lstm_cell_34/mulMul(lstm_34/while/lstm_cell_34/Sigmoid_1:y:0lstm_34_while_placeholder_3*
T0*'
_output_shapes
:         	2 
lstm_34/while/lstm_cell_34/mulз
lstm_34/while/lstm_cell_34/ReluRelu)lstm_34/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2!
lstm_34/while/lstm_cell_34/Relu╘
 lstm_34/while/lstm_cell_34/mul_1Mul&lstm_34/while/lstm_cell_34/Sigmoid:y:0-lstm_34/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/mul_1╔
 lstm_34/while/lstm_cell_34/add_1AddV2"lstm_34/while/lstm_cell_34/mul:z:0$lstm_34/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/add_1┤
$lstm_34/while/lstm_cell_34/Sigmoid_2Sigmoid)lstm_34/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2&
$lstm_34/while/lstm_cell_34/Sigmoid_2ж
!lstm_34/while/lstm_cell_34/Relu_1Relu$lstm_34/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2#
!lstm_34/while/lstm_cell_34/Relu_1╪
 lstm_34/while/lstm_cell_34/mul_2Mul(lstm_34/while/lstm_cell_34/Sigmoid_2:y:0/lstm_34/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/mul_2И
2lstm_34/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_34_while_placeholder_1lstm_34_while_placeholder$lstm_34/while/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_34/while/TensorArrayV2Write/TensorListSetIteml
lstm_34/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_34/while/add/yЙ
lstm_34/while/addAddV2lstm_34_while_placeholderlstm_34/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_34/while/addp
lstm_34/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_34/while/add_1/yЮ
lstm_34/while/add_1AddV2(lstm_34_while_lstm_34_while_loop_counterlstm_34/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_34/while/add_1Л
lstm_34/while/IdentityIdentitylstm_34/while/add_1:z:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identityж
lstm_34/while/Identity_1Identity.lstm_34_while_lstm_34_while_maximum_iterations^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_1Н
lstm_34/while/Identity_2Identitylstm_34/while/add:z:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_2║
lstm_34/while/Identity_3IdentityBlstm_34/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_3н
lstm_34/while/Identity_4Identity$lstm_34/while/lstm_cell_34/mul_2:z:0^lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_34/while/Identity_4н
lstm_34/while/Identity_5Identity$lstm_34/while/lstm_cell_34/add_1:z:0^lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_34/while/Identity_5Ж
lstm_34/while/NoOpNoOp2^lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp1^lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp3^lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_34/while/NoOp"9
lstm_34_while_identitylstm_34/while/Identity:output:0"=
lstm_34_while_identity_1!lstm_34/while/Identity_1:output:0"=
lstm_34_while_identity_2!lstm_34/while/Identity_2:output:0"=
lstm_34_while_identity_3!lstm_34/while/Identity_3:output:0"=
lstm_34_while_identity_4!lstm_34/while/Identity_4:output:0"=
lstm_34_while_identity_5!lstm_34/while/Identity_5:output:0"P
%lstm_34_while_lstm_34_strided_slice_1'lstm_34_while_lstm_34_strided_slice_1_0"z
:lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0"|
;lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0"x
9lstm_34_while_lstm_cell_34_matmul_readvariableop_resource;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0"╚
alstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensorclstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2f
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp2d
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp2h
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
л
▓
(__inference_lstm_34_layer_call_fn_322558

inputs
unknown:@$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3207002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         	2

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
ШF
В
C__inference_lstm_35_layer_call_and_return_conditional_losses_320018

inputs%
lstm_cell_35_319936:	$%
lstm_cell_35_319938:	$!
lstm_cell_35_319940:$
identityИв$lstm_cell_35/StatefulPartitionedCallвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
 :                  	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2Э
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_319936lstm_cell_35_319938lstm_cell_35_319940*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3199352&
$lstm_cell_35/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counter└
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_319936lstm_cell_35_319938lstm_cell_35_319940*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319949*
condR
while_cond_319948*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity}
NoOpNoOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
м
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_323875

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         	2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         	2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╒
├
while_cond_320780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_320780___redundant_placeholder04
0while_while_cond_320780___redundant_placeholder14
0while_while_cond_320780___redundant_placeholder24
0while_while_cond_320780___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
╫[
Ш
C__inference_lstm_34_layer_call_and_return_conditional_losses_321369

inputs=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_321285*
condR
while_cond_321284*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
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
:         	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Л?
╩
while_body_323764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
▓
d
+__inference_dropout_18_layer_call_fn_323183

inputs
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3212022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
▄[
Ш
C__inference_lstm_35_layer_call_and_return_conditional_losses_323697

inputs=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
:         	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323613*
condR
while_cond_323612*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
╥%
▌
while_body_319529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_34_319553_0:@$-
while_lstm_cell_34_319555_0:	$)
while_lstm_cell_34_319557_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_34_319553:@$+
while_lstm_cell_34_319555:	$'
while_lstm_cell_34_319557:$Ив*while/lstm_cell_34/StatefulPartitionedCall├
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
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_34_319553_0while_lstm_cell_34_319555_0while_lstm_cell_34_319557_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3194512,
*while/lstm_cell_34/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_34/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_34/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_34/StatefulPartitionedCall*"
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
while_lstm_cell_34_319553while_lstm_cell_34_319553_0"8
while_lstm_cell_34_319555while_lstm_cell_34_319555_0"8
while_lstm_cell_34_319557while_lstm_cell_34_319557_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2X
*while/lstm_cell_34/StatefulPartitionedCall*while/lstm_cell_34/StatefulPartitionedCall: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
√
ь
$__inference_signature_wrapper_321669
conv1d_6_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@$
	unknown_4:	$
	unknown_5:$
	unknown_6:	$
	unknown_7:	$
	unknown_8:$
	unknown_9:		

unknown_10:	

unknown_11:	

unknown_12:
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallconv1d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_3192022
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
ы
√
'sequential_12_lstm_34_while_cond_318946H
Dsequential_12_lstm_34_while_sequential_12_lstm_34_while_loop_counterN
Jsequential_12_lstm_34_while_sequential_12_lstm_34_while_maximum_iterations+
'sequential_12_lstm_34_while_placeholder-
)sequential_12_lstm_34_while_placeholder_1-
)sequential_12_lstm_34_while_placeholder_2-
)sequential_12_lstm_34_while_placeholder_3J
Fsequential_12_lstm_34_while_less_sequential_12_lstm_34_strided_slice_1`
\sequential_12_lstm_34_while_sequential_12_lstm_34_while_cond_318946___redundant_placeholder0`
\sequential_12_lstm_34_while_sequential_12_lstm_34_while_cond_318946___redundant_placeholder1`
\sequential_12_lstm_34_while_sequential_12_lstm_34_while_cond_318946___redundant_placeholder2`
\sequential_12_lstm_34_while_sequential_12_lstm_34_while_cond_318946___redundant_placeholder3(
$sequential_12_lstm_34_while_identity
▐
 sequential_12/lstm_34/while/LessLess'sequential_12_lstm_34_while_placeholderFsequential_12_lstm_34_while_less_sequential_12_lstm_34_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_34/while/LessЯ
$sequential_12/lstm_34/while/IdentityIdentity$sequential_12/lstm_34/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_34/while/Identity"U
$sequential_12_lstm_34_while_identity-sequential_12/lstm_34/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
н
Ў
.__inference_sequential_12_layer_call_fn_321544
conv1d_6_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@$
	unknown_4:	$
	unknown_5:$
	unknown_6:	$
	unknown_7:	$
	unknown_8:$
	unknown_9:		

unknown_10:	

unknown_11:	

unknown_12:
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallconv1d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3214802
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
╒
├
while_cond_323310
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323310___redundant_placeholder04
0while_while_cond_323310___redundant_placeholder14
0while_while_cond_323310___redundant_placeholder24
0while_while_cond_323310___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Л?
╩
while_body_322938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
┌
L
0__inference_max_pooling1d_2_layer_call_fn_322509

inputs
identity═
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
GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3205482
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
╒
├
while_cond_320158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_320158___redundant_placeholder04
0while_while_cond_320158___redundant_placeholder14
0while_while_cond_320158___redundant_placeholder24
0while_while_cond_320158___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
щ
Б
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_319305

inputs

states
states_10
matmul_readvariableop_resource:@$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_namestates:OK
'
_output_shapes
:         	
 
_user_specified_namestates
г
▓
(__inference_lstm_35_layer_call_fn_323244

inputs
unknown:	$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3211732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
┐н
╪
I__inference_sequential_12_layer_call_and_return_conditional_losses_322449

inputsJ
4conv1d_6_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_6_biasadd_readvariableop_resource: J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_7_biasadd_readvariableop_resource:@E
3lstm_34_lstm_cell_34_matmul_readvariableop_resource:@$G
5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource:	$B
4lstm_34_lstm_cell_34_biasadd_readvariableop_resource:$E
3lstm_35_lstm_cell_35_matmul_readvariableop_resource:	$G
5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	$B
4lstm_35_lstm_cell_35_biasadd_readvariableop_resource:$9
'dense_34_matmul_readvariableop_resource:		6
(dense_34_biasadd_readvariableop_resource:	9
'dense_35_matmul_readvariableop_resource:	6
(dense_35_biasadd_readvariableop_resource:
identityИвconv1d_6/BiasAdd/ReadVariableOpв+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpвconv1d_7/BiasAdd/ReadVariableOpв+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpвdense_35/BiasAdd/ReadVariableOpвdense_35/MatMul/ReadVariableOpв+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpв*lstm_34/lstm_cell_34/MatMul/ReadVariableOpв,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpвlstm_34/whileв+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpв*lstm_35/lstm_cell_35/MatMul/ReadVariableOpв,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpвlstm_35/whileЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_6/conv1d/ExpandDims/dim▒
conv1d_6/conv1d/ExpandDims
ExpandDimsinputs'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d_6/conv1d/ExpandDims╙
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim█
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_6/conv1d/ExpandDims_1┌
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv1d_6/conv1dн
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_6/conv1d/Squeezeз
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp░
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_6/BiasAddw
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_6/ReluЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_7/conv1d/ExpandDims/dim╞
conv1d_7/conv1d/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d_7/conv1d/ExpandDims╙
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim█
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_7/conv1d/ExpandDims_1┌
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv1d_7/conv1dн
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

¤        2
conv1d_7/conv1d/Squeezeз
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_7/BiasAdd/ReadVariableOp░
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:         @2
conv1d_7/ReluВ
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim╞
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @2
max_pooling1d_2/ExpandDims╬
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPoolм
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
2
max_pooling1d_2/Squeezen
lstm_34/ShapeShape max_pooling1d_2/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_34/ShapeД
lstm_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice/stackИ
lstm_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_34/strided_slice/stack_1И
lstm_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_34/strided_slice/stack_2Т
lstm_34/strided_sliceStridedSlicelstm_34/Shape:output:0$lstm_34/strided_slice/stack:output:0&lstm_34/strided_slice/stack_1:output:0&lstm_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_34/strided_slicel
lstm_34/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros/mul/yМ
lstm_34/zeros/mulMullstm_34/strided_slice:output:0lstm_34/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros/mulo
lstm_34/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_34/zeros/Less/yЗ
lstm_34/zeros/LessLesslstm_34/zeros/mul:z:0lstm_34/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros/Lessr
lstm_34/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros/packed/1г
lstm_34/zeros/packedPacklstm_34/strided_slice:output:0lstm_34/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_34/zeros/packedo
lstm_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/zeros/ConstХ
lstm_34/zerosFilllstm_34/zeros/packed:output:0lstm_34/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_34/zerosp
lstm_34/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros_1/mul/yТ
lstm_34/zeros_1/mulMullstm_34/strided_slice:output:0lstm_34/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros_1/muls
lstm_34/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_34/zeros_1/Less/yП
lstm_34/zeros_1/LessLesslstm_34/zeros_1/mul:z:0lstm_34/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros_1/Lessv
lstm_34/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros_1/packed/1й
lstm_34/zeros_1/packedPacklstm_34/strided_slice:output:0!lstm_34/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_34/zeros_1/packeds
lstm_34/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/zeros_1/ConstЭ
lstm_34/zeros_1Filllstm_34/zeros_1/packed:output:0lstm_34/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_34/zeros_1Е
lstm_34/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_34/transpose/permм
lstm_34/transpose	Transpose max_pooling1d_2/Squeeze:output:0lstm_34/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_34/transposeg
lstm_34/Shape_1Shapelstm_34/transpose:y:0*
T0*
_output_shapes
:2
lstm_34/Shape_1И
lstm_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice_1/stackМ
lstm_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_1/stack_1М
lstm_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_1/stack_2Ю
lstm_34/strided_slice_1StridedSlicelstm_34/Shape_1:output:0&lstm_34/strided_slice_1/stack:output:0(lstm_34/strided_slice_1/stack_1:output:0(lstm_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_34/strided_slice_1Х
#lstm_34/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_34/TensorArrayV2/element_shape╥
lstm_34/TensorArrayV2TensorListReserve,lstm_34/TensorArrayV2/element_shape:output:0 lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_34/TensorArrayV2╧
=lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2?
=lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_34/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_34/transpose:y:0Flstm_34/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_34/TensorArrayUnstack/TensorListFromTensorИ
lstm_34/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice_2/stackМ
lstm_34/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_2/stack_1М
lstm_34/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_2/stack_2м
lstm_34/strided_slice_2StridedSlicelstm_34/transpose:y:0&lstm_34/strided_slice_2/stack:output:0(lstm_34/strided_slice_2/stack_1:output:0(lstm_34/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_34/strided_slice_2╠
*lstm_34/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3lstm_34_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02,
*lstm_34/lstm_cell_34/MatMul/ReadVariableOp╠
lstm_34/lstm_cell_34/MatMulMatMul lstm_34/strided_slice_2:output:02lstm_34/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/MatMul╥
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02.
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp╚
lstm_34/lstm_cell_34/MatMul_1MatMullstm_34/zeros:output:04lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/MatMul_1┐
lstm_34/lstm_cell_34/addAddV2%lstm_34/lstm_cell_34/MatMul:product:0'lstm_34/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/add╦
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp╠
lstm_34/lstm_cell_34/BiasAddBiasAddlstm_34/lstm_cell_34/add:z:03lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/BiasAddО
$lstm_34/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_34/lstm_cell_34/split/split_dimУ
lstm_34/lstm_cell_34/splitSplit-lstm_34/lstm_cell_34/split/split_dim:output:0%lstm_34/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_34/lstm_cell_34/splitЮ
lstm_34/lstm_cell_34/SigmoidSigmoid#lstm_34/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Sigmoidв
lstm_34/lstm_cell_34/Sigmoid_1Sigmoid#lstm_34/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2 
lstm_34/lstm_cell_34/Sigmoid_1л
lstm_34/lstm_cell_34/mulMul"lstm_34/lstm_cell_34/Sigmoid_1:y:0lstm_34/zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mulХ
lstm_34/lstm_cell_34/ReluRelu#lstm_34/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Relu╝
lstm_34/lstm_cell_34/mul_1Mul lstm_34/lstm_cell_34/Sigmoid:y:0'lstm_34/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mul_1▒
lstm_34/lstm_cell_34/add_1AddV2lstm_34/lstm_cell_34/mul:z:0lstm_34/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/add_1в
lstm_34/lstm_cell_34/Sigmoid_2Sigmoid#lstm_34/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2 
lstm_34/lstm_cell_34/Sigmoid_2Ф
lstm_34/lstm_cell_34/Relu_1Relulstm_34/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Relu_1└
lstm_34/lstm_cell_34/mul_2Mul"lstm_34/lstm_cell_34/Sigmoid_2:y:0)lstm_34/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mul_2Я
%lstm_34/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2'
%lstm_34/TensorArrayV2_1/element_shape╪
lstm_34/TensorArrayV2_1TensorListReserve.lstm_34/TensorArrayV2_1/element_shape:output:0 lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_34/TensorArrayV2_1^
lstm_34/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_34/timeП
 lstm_34/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_34/while/maximum_iterationsz
lstm_34/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_34/while/loop_counterГ
lstm_34/whileWhile#lstm_34/while/loop_counter:output:0)lstm_34/while/maximum_iterations:output:0lstm_34/time:output:0 lstm_34/TensorArrayV2_1:handle:0lstm_34/zeros:output:0lstm_34/zeros_1:output:0 lstm_34/strided_slice_1:output:0?lstm_34/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_34_lstm_cell_34_matmul_readvariableop_resource5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource4lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_34_while_body_322180*%
condR
lstm_34_while_cond_322179*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
lstm_34/while┼
8lstm_34/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2:
8lstm_34/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_34/TensorArrayV2Stack/TensorListStackTensorListStacklstm_34/while:output:3Alstm_34/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02,
*lstm_34/TensorArrayV2Stack/TensorListStackС
lstm_34/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_34/strided_slice_3/stackМ
lstm_34/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_34/strided_slice_3/stack_1М
lstm_34/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_3/stack_2╩
lstm_34/strided_slice_3StridedSlice3lstm_34/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_34/strided_slice_3/stack:output:0(lstm_34/strided_slice_3/stack_1:output:0(lstm_34/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_34/strided_slice_3Й
lstm_34/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_34/transpose_1/perm┼
lstm_34/transpose_1	Transpose3lstm_34/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_34/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_34/transpose_1v
lstm_34/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/runtimey
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_18/dropout/Constй
dropout_18/dropout/MulMullstm_34/transpose_1:y:0!dropout_18/dropout/Const:output:0*
T0*+
_output_shapes
:         	2
dropout_18/dropout/Mul{
dropout_18/dropout/ShapeShapelstm_34/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape┘
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*+
_output_shapes
:         	*
dtype021
/dropout_18/dropout/random_uniform/RandomUniformЛ
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_18/dropout/GreaterEqual/yю
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         	2!
dropout_18/dropout/GreaterEqualд
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	2
dropout_18/dropout/Castк
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*+
_output_shapes
:         	2
dropout_18/dropout/Mul_1j
lstm_35/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_35/ShapeД
lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice/stackИ
lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_35/strided_slice/stack_1И
lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_35/strided_slice/stack_2Т
lstm_35/strided_sliceStridedSlicelstm_35/Shape:output:0$lstm_35/strided_slice/stack:output:0&lstm_35/strided_slice/stack_1:output:0&lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_35/strided_slicel
lstm_35/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros/mul/yМ
lstm_35/zeros/mulMullstm_35/strided_slice:output:0lstm_35/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros/mulo
lstm_35/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_35/zeros/Less/yЗ
lstm_35/zeros/LessLesslstm_35/zeros/mul:z:0lstm_35/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros/Lessr
lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros/packed/1г
lstm_35/zeros/packedPacklstm_35/strided_slice:output:0lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_35/zeros/packedo
lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/zeros/ConstХ
lstm_35/zerosFilllstm_35/zeros/packed:output:0lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_35/zerosp
lstm_35/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros_1/mul/yТ
lstm_35/zeros_1/mulMullstm_35/strided_slice:output:0lstm_35/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros_1/muls
lstm_35/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_35/zeros_1/Less/yП
lstm_35/zeros_1/LessLesslstm_35/zeros_1/mul:z:0lstm_35/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros_1/Lessv
lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros_1/packed/1й
lstm_35/zeros_1/packedPacklstm_35/strided_slice:output:0!lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_35/zeros_1/packeds
lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/zeros_1/ConstЭ
lstm_35/zeros_1Filllstm_35/zeros_1/packed:output:0lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_35/zeros_1Е
lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_35/transpose/permи
lstm_35/transpose	Transposedropout_18/dropout/Mul_1:z:0lstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_35/transposeg
lstm_35/Shape_1Shapelstm_35/transpose:y:0*
T0*
_output_shapes
:2
lstm_35/Shape_1И
lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice_1/stackМ
lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_1/stack_1М
lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_1/stack_2Ю
lstm_35/strided_slice_1StridedSlicelstm_35/Shape_1:output:0&lstm_35/strided_slice_1/stack:output:0(lstm_35/strided_slice_1/stack_1:output:0(lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_35/strided_slice_1Х
#lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_35/TensorArrayV2/element_shape╥
lstm_35/TensorArrayV2TensorListReserve,lstm_35/TensorArrayV2/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_35/TensorArrayV2╧
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2?
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_35/transpose:y:0Flstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_35/TensorArrayUnstack/TensorListFromTensorИ
lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice_2/stackМ
lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_2/stack_1М
lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_2/stack_2м
lstm_35/strided_slice_2StridedSlicelstm_35/transpose:y:0&lstm_35/strided_slice_2/stack:output:0(lstm_35/strided_slice_2/stack_1:output:0(lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_35/strided_slice_2╠
*lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02,
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp╠
lstm_35/lstm_cell_35/MatMulMatMul lstm_35/strided_slice_2:output:02lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/MatMul╥
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02.
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp╚
lstm_35/lstm_cell_35/MatMul_1MatMullstm_35/zeros:output:04lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/MatMul_1┐
lstm_35/lstm_cell_35/addAddV2%lstm_35/lstm_cell_35/MatMul:product:0'lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/add╦
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp╠
lstm_35/lstm_cell_35/BiasAddBiasAddlstm_35/lstm_cell_35/add:z:03lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/BiasAddО
$lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_35/lstm_cell_35/split/split_dimУ
lstm_35/lstm_cell_35/splitSplit-lstm_35/lstm_cell_35/split/split_dim:output:0%lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_35/lstm_cell_35/splitЮ
lstm_35/lstm_cell_35/SigmoidSigmoid#lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Sigmoidв
lstm_35/lstm_cell_35/Sigmoid_1Sigmoid#lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2 
lstm_35/lstm_cell_35/Sigmoid_1л
lstm_35/lstm_cell_35/mulMul"lstm_35/lstm_cell_35/Sigmoid_1:y:0lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mulХ
lstm_35/lstm_cell_35/ReluRelu#lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Relu╝
lstm_35/lstm_cell_35/mul_1Mul lstm_35/lstm_cell_35/Sigmoid:y:0'lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mul_1▒
lstm_35/lstm_cell_35/add_1AddV2lstm_35/lstm_cell_35/mul:z:0lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/add_1в
lstm_35/lstm_cell_35/Sigmoid_2Sigmoid#lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2 
lstm_35/lstm_cell_35/Sigmoid_2Ф
lstm_35/lstm_cell_35/Relu_1Relulstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Relu_1└
lstm_35/lstm_cell_35/mul_2Mul"lstm_35/lstm_cell_35/Sigmoid_2:y:0)lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mul_2Я
%lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2'
%lstm_35/TensorArrayV2_1/element_shape╪
lstm_35/TensorArrayV2_1TensorListReserve.lstm_35/TensorArrayV2_1/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_35/TensorArrayV2_1^
lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_35/timeП
 lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_35/while/maximum_iterationsz
lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_35/while/loop_counterГ
lstm_35/whileWhile#lstm_35/while/loop_counter:output:0)lstm_35/while/maximum_iterations:output:0lstm_35/time:output:0 lstm_35/TensorArrayV2_1:handle:0lstm_35/zeros:output:0lstm_35/zeros_1:output:0 lstm_35/strided_slice_1:output:0?lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_35_lstm_cell_35_matmul_readvariableop_resource5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_35_while_body_322335*%
condR
lstm_35_while_cond_322334*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
lstm_35/while┼
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2:
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_35/TensorArrayV2Stack/TensorListStackTensorListStacklstm_35/while:output:3Alstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02,
*lstm_35/TensorArrayV2Stack/TensorListStackС
lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_35/strided_slice_3/stackМ
lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_35/strided_slice_3/stack_1М
lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_3/stack_2╩
lstm_35/strided_slice_3StridedSlice3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_35/strided_slice_3/stack:output:0(lstm_35/strided_slice_3/stack_1:output:0(lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_35/strided_slice_3Й
lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_35/transpose_1/perm┼
lstm_35/transpose_1	Transpose3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_35/transpose_1v
lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/runtimey
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_19/dropout/Constо
dropout_19/dropout/MulMul lstm_35/strided_slice_3:output:0!dropout_19/dropout/Const:output:0*
T0*'
_output_shapes
:         	2
dropout_19/dropout/MulД
dropout_19/dropout/ShapeShape lstm_35/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape╒
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*'
_output_shapes
:         	*
dtype021
/dropout_19/dropout/random_uniform/RandomUniformЛ
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_19/dropout/GreaterEqual/yъ
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         	2!
dropout_19/dropout/GreaterEqualа
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         	2
dropout_19/dropout/Castж
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*'
_output_shapes
:         	2
dropout_19/dropout/Mul_1и
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02 
dense_34/MatMul/ReadVariableOpд
dense_34/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
dense_34/MatMulз
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_34/BiasAdd/ReadVariableOpе
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
dense_34/Reluи
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02 
dense_35/MatMul/ReadVariableOpг
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_35/MatMulз
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOpе
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_35/BiasAddm
reshape_17/ShapeShapedense_35/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_17/ShapeК
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stackО
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1О
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2д
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2╫
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shapeз
reshape_17/ReshapeReshapedense_35/BiasAdd:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_17/Reshapez
IdentityIdentityreshape_17/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp,^lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp+^lstm_34/lstm_cell_34/MatMul/ReadVariableOp-^lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp^lstm_34/while,^lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+^lstm_35/lstm_cell_35/MatMul/ReadVariableOp-^lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp^lstm_35/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2Z
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp2X
*lstm_34/lstm_cell_34/MatMul/ReadVariableOp*lstm_34/lstm_cell_34/MatMul/ReadVariableOp2\
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp2
lstm_34/whilelstm_34/while2Z
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2X
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp*lstm_35/lstm_cell_35/MatMul/ReadVariableOp2\
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2
lstm_35/whilelstm_35/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╓
┤
(__inference_lstm_34_layer_call_fn_322536
inputs_0
unknown:@$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3193882
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  	2

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
╫[
Ш
C__inference_lstm_34_layer_call_and_return_conditional_losses_323022

inputs=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322938*
condR
while_cond_322937*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
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
:         	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
є
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_323863

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         	2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╦2
▐
I__inference_sequential_12_layer_call_and_return_conditional_losses_321628
conv1d_6_input%
conv1d_6_321589: 
conv1d_6_321591: %
conv1d_7_321594: @
conv1d_7_321596:@ 
lstm_34_321600:@$ 
lstm_34_321602:	$
lstm_34_321604:$ 
lstm_35_321608:	$ 
lstm_35_321610:	$
lstm_35_321612:$!
dense_34_321616:		
dense_34_321618:	!
dense_35_321621:	
dense_35_321623:
identityИв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCallвlstm_34/StatefulPartitionedCallвlstm_35/StatefulPartitionedCallа
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallconv1d_6_inputconv1d_6_321589conv1d_6_321591*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3205132"
 conv1d_6/StatefulPartitionedCall╗
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0conv1d_7_321594conv1d_7_321596*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3205352"
 conv1d_7/StatefulPartitionedCallР
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3205482!
max_pooling1d_2/PartitionedCall╟
lstm_34/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0lstm_34_321600lstm_34_321602lstm_34_321604*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3213692!
lstm_34/StatefulPartitionedCallШ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3212022$
"dropout_18/StatefulPartitionedCall╞
lstm_35/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_35_321608lstm_35_321610lstm_35_321612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3211732!
lstm_35/StatefulPartitionedCall╣
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3210062$
"dropout_19/StatefulPartitionedCall╣
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_34_321616dense_34_321618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3208912"
 dense_34/StatefulPartitionedCall╖
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_321621dense_35_321623*
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
GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_3209072"
 dense_35/StatefulPartitionedCallБ
reshape_17/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *O
fJRH
F__inference_reshape_17_layer_call_and_return_conditional_losses_3209262
reshape_17/PartitionedCallВ
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityш
NoOpNoOp!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_34/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_34/StatefulPartitionedCalllstm_34/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
ё
Г
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324096

inputs
states_0
states_10
matmul_readvariableop_resource:	$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         	
 
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
╗
┤
(__inference_lstm_35_layer_call_fn_323211
inputs_0
unknown:	$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3200182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
╒
├
while_cond_323612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323612___redundant_placeholder04
0while_while_cond_323612___redundant_placeholder14
0while_while_cond_323612___redundant_placeholder24
0while_while_cond_323612___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
╞

у
lstm_35_while_cond_321977,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3.
*lstm_35_while_less_lstm_35_strided_slice_1D
@lstm_35_while_lstm_35_while_cond_321977___redundant_placeholder0D
@lstm_35_while_lstm_35_while_cond_321977___redundant_placeholder1D
@lstm_35_while_lstm_35_while_cond_321977___redundant_placeholder2D
@lstm_35_while_lstm_35_while_cond_321977___redundant_placeholder3
lstm_35_while_identity
Ш
lstm_35/while/LessLesslstm_35_while_placeholder*lstm_35_while_less_lstm_35_strided_slice_1*
T0*
_output_shapes
: 2
lstm_35/while/Lessu
lstm_35/while/IdentityIdentitylstm_35/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_35/while/Identity"9
lstm_35_while_identitylstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Л?
╩
while_body_322787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
└J
╩

lstm_35_while_body_321978,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3+
'lstm_35_while_lstm_35_strided_slice_1_0g
clstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	$O
=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$J
<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:$
lstm_35_while_identity
lstm_35_while_identity_1
lstm_35_while_identity_2
lstm_35_while_identity_3
lstm_35_while_identity_4
lstm_35_while_identity_5)
%lstm_35_while_lstm_35_strided_slice_1e
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorK
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	$M
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	$H
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:$Ив1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpв0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpв2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp╙
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2A
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0lstm_35_while_placeholderHlstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype023
1lstm_35/while/TensorArrayV2Read/TensorListGetItemр
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype022
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpЎ
!lstm_35/while/lstm_cell_35/MatMulMatMul8lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2#
!lstm_35/while/lstm_cell_35/MatMulц
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype024
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp▀
#lstm_35/while/lstm_cell_35/MatMul_1MatMullstm_35_while_placeholder_2:lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2%
#lstm_35/while/lstm_cell_35/MatMul_1╫
lstm_35/while/lstm_cell_35/addAddV2+lstm_35/while/lstm_cell_35/MatMul:product:0-lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2 
lstm_35/while/lstm_cell_35/add▀
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype023
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpф
"lstm_35/while/lstm_cell_35/BiasAddBiasAdd"lstm_35/while/lstm_cell_35/add:z:09lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2$
"lstm_35/while/lstm_cell_35/BiasAddЪ
*lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_35/while/lstm_cell_35/split/split_dimл
 lstm_35/while/lstm_cell_35/splitSplit3lstm_35/while/lstm_cell_35/split/split_dim:output:0+lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2"
 lstm_35/while/lstm_cell_35/split░
"lstm_35/while/lstm_cell_35/SigmoidSigmoid)lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2$
"lstm_35/while/lstm_cell_35/Sigmoid┤
$lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid)lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2&
$lstm_35/while/lstm_cell_35/Sigmoid_1└
lstm_35/while/lstm_cell_35/mulMul(lstm_35/while/lstm_cell_35/Sigmoid_1:y:0lstm_35_while_placeholder_3*
T0*'
_output_shapes
:         	2 
lstm_35/while/lstm_cell_35/mulз
lstm_35/while/lstm_cell_35/ReluRelu)lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2!
lstm_35/while/lstm_cell_35/Relu╘
 lstm_35/while/lstm_cell_35/mul_1Mul&lstm_35/while/lstm_cell_35/Sigmoid:y:0-lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/mul_1╔
 lstm_35/while/lstm_cell_35/add_1AddV2"lstm_35/while/lstm_cell_35/mul:z:0$lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/add_1┤
$lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid)lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2&
$lstm_35/while/lstm_cell_35/Sigmoid_2ж
!lstm_35/while/lstm_cell_35/Relu_1Relu$lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2#
!lstm_35/while/lstm_cell_35/Relu_1╪
 lstm_35/while/lstm_cell_35/mul_2Mul(lstm_35/while/lstm_cell_35/Sigmoid_2:y:0/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/mul_2И
2lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_35_while_placeholder_1lstm_35_while_placeholder$lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_35/while/TensorArrayV2Write/TensorListSetIteml
lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_35/while/add/yЙ
lstm_35/while/addAddV2lstm_35_while_placeholderlstm_35/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_35/while/addp
lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_35/while/add_1/yЮ
lstm_35/while/add_1AddV2(lstm_35_while_lstm_35_while_loop_counterlstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_35/while/add_1Л
lstm_35/while/IdentityIdentitylstm_35/while/add_1:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identityж
lstm_35/while/Identity_1Identity.lstm_35_while_lstm_35_while_maximum_iterations^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_1Н
lstm_35/while/Identity_2Identitylstm_35/while/add:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_2║
lstm_35/while/Identity_3IdentityBlstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_3н
lstm_35/while/Identity_4Identity$lstm_35/while/lstm_cell_35/mul_2:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_35/while/Identity_4н
lstm_35/while/Identity_5Identity$lstm_35/while/lstm_cell_35/add_1:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_35/while/Identity_5Ж
lstm_35/while/NoOpNoOp2^lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1^lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp3^lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_35/while/NoOp"9
lstm_35_while_identitylstm_35/while/Identity:output:0"=
lstm_35_while_identity_1!lstm_35/while/Identity_1:output:0"=
lstm_35_while_identity_2!lstm_35/while/Identity_2:output:0"=
lstm_35_while_identity_3!lstm_35/while/Identity_3:output:0"=
lstm_35_while_identity_4!lstm_35/while/Identity_4:output:0"=
lstm_35_while_identity_5!lstm_35/while/Identity_5:output:0"P
%lstm_35_while_lstm_35_strided_slice_1'lstm_35_while_lstm_35_strided_slice_1_0"z
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"|
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"x
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"╚
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2f
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2d
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2h
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
Ц\
Ъ
C__inference_lstm_34_layer_call_and_return_conditional_losses_322871
inputs_0=
+lstm_cell_34_matmul_readvariableop_resource:@$?
-lstm_cell_34_matmul_1_readvariableop_resource:	$:
,lstm_cell_34_biasadd_readvariableop_resource:$
identityИв#lstm_cell_34/BiasAdd/ReadVariableOpв"lstm_cell_34/MatMul/ReadVariableOpв$lstm_cell_34/MatMul_1/ReadVariableOpвwhileF
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2┤
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOpм
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul║
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOpи
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/MatMul_1Я
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/add│
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOpм
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dimє
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_34/splitЖ
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/SigmoidК
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_1Л
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_34/ReluЬ
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_1С
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/add_1К
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/Relu_1а
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_34/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322787*
condR
while_cond_322786*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
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
 :                  	2

Identity╚
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
ж/
М
I__inference_sequential_12_layer_call_and_return_conditional_losses_320929

inputs%
conv1d_6_320514: 
conv1d_6_320516: %
conv1d_7_320536: @
conv1d_7_320538:@ 
lstm_34_320701:@$ 
lstm_34_320703:	$
lstm_34_320705:$ 
lstm_35_320866:	$ 
lstm_35_320868:	$
lstm_35_320870:$!
dense_34_320892:		
dense_34_320894:	!
dense_35_320908:	
dense_35_320910:
identityИв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallвlstm_34/StatefulPartitionedCallвlstm_35/StatefulPartitionedCallШ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_6_320514conv1d_6_320516*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3205132"
 conv1d_6/StatefulPartitionedCall╗
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0conv1d_7_320536conv1d_7_320538*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3205352"
 conv1d_7/StatefulPartitionedCallР
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3205482!
max_pooling1d_2/PartitionedCall╟
lstm_34/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0lstm_34_320701lstm_34_320703lstm_34_320705*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3207002!
lstm_34/StatefulPartitionedCallА
dropout_18/PartitionedCallPartitionedCall(lstm_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3207132
dropout_18/PartitionedCall╛
lstm_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_35_320866lstm_35_320868lstm_35_320870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3208652!
lstm_35/StatefulPartitionedCall№
dropout_19/PartitionedCallPartitionedCall(lstm_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3208782
dropout_19/PartitionedCall▒
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_34_320892dense_34_320894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3208912"
 dense_34/StatefulPartitionedCall╖
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_320908dense_35_320910*
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
GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_3209072"
 dense_35/StatefulPartitionedCallБ
reshape_17/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *O
fJRH
F__inference_reshape_17_layer_call_and_return_conditional_losses_3209262
reshape_17/PartitionedCallВ
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityЮ
NoOpNoOp!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^lstm_34/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
lstm_34/StatefulPartitionedCalllstm_34/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Е
Ъ
)__inference_conv1d_7_layer_call_fn_322483

inputs
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3205352
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
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
┐^
М
'sequential_12_lstm_34_while_body_318947H
Dsequential_12_lstm_34_while_sequential_12_lstm_34_while_loop_counterN
Jsequential_12_lstm_34_while_sequential_12_lstm_34_while_maximum_iterations+
'sequential_12_lstm_34_while_placeholder-
)sequential_12_lstm_34_while_placeholder_1-
)sequential_12_lstm_34_while_placeholder_2-
)sequential_12_lstm_34_while_placeholder_3G
Csequential_12_lstm_34_while_sequential_12_lstm_34_strided_slice_1_0Г
sequential_12_lstm_34_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_34_tensorarrayunstack_tensorlistfromtensor_0[
Isequential_12_lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0:@$]
Ksequential_12_lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$X
Jsequential_12_lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0:$(
$sequential_12_lstm_34_while_identity*
&sequential_12_lstm_34_while_identity_1*
&sequential_12_lstm_34_while_identity_2*
&sequential_12_lstm_34_while_identity_3*
&sequential_12_lstm_34_while_identity_4*
&sequential_12_lstm_34_while_identity_5E
Asequential_12_lstm_34_while_sequential_12_lstm_34_strided_slice_1Б
}sequential_12_lstm_34_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_34_tensorarrayunstack_tensorlistfromtensorY
Gsequential_12_lstm_34_while_lstm_cell_34_matmul_readvariableop_resource:@$[
Isequential_12_lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource:	$V
Hsequential_12_lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource:$Ив?sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpв>sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpв@sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOpя
Msequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2O
Msequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_34_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_34_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_34_while_placeholderVsequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02A
?sequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItemК
>sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02@
>sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpо
/sequential_12/lstm_34/while/lstm_cell_34/MatMulMatMulFsequential_12/lstm_34/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $21
/sequential_12/lstm_34/while/lstm_cell_34/MatMulР
@sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02B
@sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOpЧ
1sequential_12/lstm_34/while/lstm_cell_34/MatMul_1MatMul)sequential_12_lstm_34_while_placeholder_2Hsequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $23
1sequential_12/lstm_34/while/lstm_cell_34/MatMul_1П
,sequential_12/lstm_34/while/lstm_cell_34/addAddV29sequential_12/lstm_34/while/lstm_cell_34/MatMul:product:0;sequential_12/lstm_34/while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2.
,sequential_12/lstm_34/while/lstm_cell_34/addЙ
?sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02A
?sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpЬ
0sequential_12/lstm_34/while/lstm_cell_34/BiasAddBiasAdd0sequential_12/lstm_34/while/lstm_cell_34/add:z:0Gsequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $22
0sequential_12/lstm_34/while/lstm_cell_34/BiasAdd╢
8sequential_12/lstm_34/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_34/while/lstm_cell_34/split/split_dimу
.sequential_12/lstm_34/while/lstm_cell_34/splitSplitAsequential_12/lstm_34/while/lstm_cell_34/split/split_dim:output:09sequential_12/lstm_34/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split20
.sequential_12/lstm_34/while/lstm_cell_34/split┌
0sequential_12/lstm_34/while/lstm_cell_34/SigmoidSigmoid7sequential_12/lstm_34/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	22
0sequential_12/lstm_34/while/lstm_cell_34/Sigmoid▐
2sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_1Sigmoid7sequential_12/lstm_34/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	24
2sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_1°
,sequential_12/lstm_34/while/lstm_cell_34/mulMul6sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_1:y:0)sequential_12_lstm_34_while_placeholder_3*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_34/while/lstm_cell_34/mul╤
-sequential_12/lstm_34/while/lstm_cell_34/ReluRelu7sequential_12/lstm_34/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2/
-sequential_12/lstm_34/while/lstm_cell_34/ReluМ
.sequential_12/lstm_34/while/lstm_cell_34/mul_1Mul4sequential_12/lstm_34/while/lstm_cell_34/Sigmoid:y:0;sequential_12/lstm_34/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_34/while/lstm_cell_34/mul_1Б
.sequential_12/lstm_34/while/lstm_cell_34/add_1AddV20sequential_12/lstm_34/while/lstm_cell_34/mul:z:02sequential_12/lstm_34/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_34/while/lstm_cell_34/add_1▐
2sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_2Sigmoid7sequential_12/lstm_34/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	24
2sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_2╨
/sequential_12/lstm_34/while/lstm_cell_34/Relu_1Relu2sequential_12/lstm_34/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	21
/sequential_12/lstm_34/while/lstm_cell_34/Relu_1Р
.sequential_12/lstm_34/while/lstm_cell_34/mul_2Mul6sequential_12/lstm_34/while/lstm_cell_34/Sigmoid_2:y:0=sequential_12/lstm_34/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	20
.sequential_12/lstm_34/while/lstm_cell_34/mul_2╬
@sequential_12/lstm_34/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_34_while_placeholder_1'sequential_12_lstm_34_while_placeholder2sequential_12/lstm_34/while/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_34/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_12/lstm_34/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_34/while/add/y┴
sequential_12/lstm_34/while/addAddV2'sequential_12_lstm_34_while_placeholder*sequential_12/lstm_34/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_34/while/addМ
#sequential_12/lstm_34/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_34/while/add_1/yф
!sequential_12/lstm_34/while/add_1AddV2Dsequential_12_lstm_34_while_sequential_12_lstm_34_while_loop_counter,sequential_12/lstm_34/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_34/while/add_1├
$sequential_12/lstm_34/while/IdentityIdentity%sequential_12/lstm_34/while/add_1:z:0!^sequential_12/lstm_34/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_34/while/Identityь
&sequential_12/lstm_34/while/Identity_1IdentityJsequential_12_lstm_34_while_sequential_12_lstm_34_while_maximum_iterations!^sequential_12/lstm_34/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_34/while/Identity_1┼
&sequential_12/lstm_34/while/Identity_2Identity#sequential_12/lstm_34/while/add:z:0!^sequential_12/lstm_34/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_34/while/Identity_2Є
&sequential_12/lstm_34/while/Identity_3IdentityPsequential_12/lstm_34/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_34/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_34/while/Identity_3х
&sequential_12/lstm_34/while/Identity_4Identity2sequential_12/lstm_34/while/lstm_cell_34/mul_2:z:0!^sequential_12/lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_34/while/Identity_4х
&sequential_12/lstm_34/while/Identity_5Identity2sequential_12/lstm_34/while/lstm_cell_34/add_1:z:0!^sequential_12/lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_34/while/Identity_5╠
 sequential_12/lstm_34/while/NoOpNoOp@^sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp?^sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpA^sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_34/while/NoOp"U
$sequential_12_lstm_34_while_identity-sequential_12/lstm_34/while/Identity:output:0"Y
&sequential_12_lstm_34_while_identity_1/sequential_12/lstm_34/while/Identity_1:output:0"Y
&sequential_12_lstm_34_while_identity_2/sequential_12/lstm_34/while/Identity_2:output:0"Y
&sequential_12_lstm_34_while_identity_3/sequential_12/lstm_34/while/Identity_3:output:0"Y
&sequential_12_lstm_34_while_identity_4/sequential_12/lstm_34/while/Identity_4:output:0"Y
&sequential_12_lstm_34_while_identity_5/sequential_12/lstm_34/while/Identity_5:output:0"Ц
Hsequential_12_lstm_34_while_lstm_cell_34_biasadd_readvariableop_resourceJsequential_12_lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0"Ш
Isequential_12_lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resourceKsequential_12_lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0"Ф
Gsequential_12_lstm_34_while_lstm_cell_34_matmul_readvariableop_resourceIsequential_12_lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0"И
Asequential_12_lstm_34_while_sequential_12_lstm_34_strided_slice_1Csequential_12_lstm_34_while_sequential_12_lstm_34_strided_slice_1_0"А
}sequential_12_lstm_34_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_34_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_34_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_34_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2В
?sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp?sequential_12/lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp2А
>sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp>sequential_12/lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp2Д
@sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp@sequential_12/lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╗
┤
(__inference_lstm_35_layer_call_fn_323222
inputs_0
unknown:	$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3202282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
в
d
+__inference_dropout_19_layer_call_fn_323858

inputs
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3210062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╒
├
while_cond_319318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_319318___redundant_placeholder04
0while_while_cond_319318___redundant_placeholder14
0while_while_cond_319318___redundant_placeholder24
0while_while_cond_319318___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
яh
м
__inference__traced_save_324298
file_prefix.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_34_lstm_cell_34_kernel_read_readvariableopD
@savev2_lstm_34_lstm_cell_34_recurrent_kernel_read_readvariableop8
4savev2_lstm_34_lstm_cell_34_bias_read_readvariableop:
6savev2_lstm_35_lstm_cell_35_kernel_read_readvariableopD
@savev2_lstm_35_lstm_cell_35_recurrent_kernel_read_readvariableop8
4savev2_lstm_35_lstm_cell_35_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv1d_6_kernel_m_read_readvariableop3
/savev2_adam_conv1d_6_bias_m_read_readvariableop5
1savev2_adam_conv1d_7_kernel_m_read_readvariableop3
/savev2_adam_conv1d_7_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableopA
=savev2_adam_lstm_34_lstm_cell_34_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_34_lstm_cell_34_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_34_lstm_cell_34_bias_m_read_readvariableopA
=savev2_adam_lstm_35_lstm_cell_35_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_35_lstm_cell_35_bias_m_read_readvariableop5
1savev2_adam_conv1d_6_kernel_v_read_readvariableop3
/savev2_adam_conv1d_6_bias_v_read_readvariableop5
1savev2_adam_conv1d_7_kernel_v_read_readvariableop3
/savev2_adam_conv1d_7_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableopA
=savev2_adam_lstm_34_lstm_cell_34_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_34_lstm_cell_34_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_34_lstm_cell_34_bias_v_read_readvariableopA
=savev2_adam_lstm_35_lstm_cell_35_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_35_lstm_cell_35_bias_v_read_readvariableop
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
ShardedFilenameо
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*└
value╢B│2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesь
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_34_lstm_cell_34_kernel_read_readvariableop@savev2_lstm_34_lstm_cell_34_recurrent_kernel_read_readvariableop4savev2_lstm_34_lstm_cell_34_bias_read_readvariableop6savev2_lstm_35_lstm_cell_35_kernel_read_readvariableop@savev2_lstm_35_lstm_cell_35_recurrent_kernel_read_readvariableop4savev2_lstm_35_lstm_cell_35_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv1d_6_kernel_m_read_readvariableop/savev2_adam_conv1d_6_bias_m_read_readvariableop1savev2_adam_conv1d_7_kernel_m_read_readvariableop/savev2_adam_conv1d_7_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop=savev2_adam_lstm_34_lstm_cell_34_kernel_m_read_readvariableopGsavev2_adam_lstm_34_lstm_cell_34_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_34_lstm_cell_34_bias_m_read_readvariableop=savev2_adam_lstm_35_lstm_cell_35_kernel_m_read_readvariableopGsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_35_lstm_cell_35_bias_m_read_readvariableop1savev2_adam_conv1d_6_kernel_v_read_readvariableop/savev2_adam_conv1d_6_bias_v_read_readvariableop1savev2_adam_conv1d_7_kernel_v_read_readvariableop/savev2_adam_conv1d_7_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop=savev2_adam_lstm_34_lstm_cell_34_kernel_v_read_readvariableopGsavev2_adam_lstm_34_lstm_cell_34_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_34_lstm_cell_34_bias_v_read_readvariableop=savev2_adam_lstm_35_lstm_cell_35_kernel_v_read_readvariableopGsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_35_lstm_cell_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*Ы
_input_shapesЙ
Ж: : : : @:@:		:	:	:: : : : : :@$:	$:$:	$:	$:$: : : : : @:@:		:	:	::@$:	$:$:	$:	$:$: : : @:@:		:	:	::@$:	$:$:	$:	$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@$:$ 

_output_shapes

:	$: 

_output_shapes
:$:$ 

_output_shapes

:	$:$ 

_output_shapes

:	$: 

_output_shapes
:$:

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:@$:$ 

_output_shapes

:	$:  

_output_shapes
:$:$! 

_output_shapes

:	$:$" 

_output_shapes

:	$: #

_output_shapes
:$:($$
"
_output_shapes
: : %

_output_shapes
: :(&$
"
_output_shapes
: @: '

_output_shapes
:@:$( 

_output_shapes

:		: )

_output_shapes
:	:$* 

_output_shapes

:	: +

_output_shapes
::$, 

_output_shapes

:@$:$- 

_output_shapes

:	$: .

_output_shapes
:$:$/ 

_output_shapes

:	$:$0 

_output_shapes

:	$: 1

_output_shapes
:$:2

_output_shapes
: 
л
У
D__inference_conv1d_6_layer_call_and_return_conditional_losses_320513

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╢
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:          2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
л
У
D__inference_conv1d_6_layer_call_and_return_conditional_losses_322474

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╢
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:          2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
│
є
-__inference_lstm_cell_34_layer_call_fn_323949

inputs
states_0
states_1
unknown:@$
	unknown_0:	$
	unknown_1:$
identity

identity_1

identity_2ИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3193052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         	2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
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
пЪ
╪
I__inference_sequential_12_layer_call_and_return_conditional_losses_322085

inputsJ
4conv1d_6_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_6_biasadd_readvariableop_resource: J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_7_biasadd_readvariableop_resource:@E
3lstm_34_lstm_cell_34_matmul_readvariableop_resource:@$G
5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource:	$B
4lstm_34_lstm_cell_34_biasadd_readvariableop_resource:$E
3lstm_35_lstm_cell_35_matmul_readvariableop_resource:	$G
5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	$B
4lstm_35_lstm_cell_35_biasadd_readvariableop_resource:$9
'dense_34_matmul_readvariableop_resource:		6
(dense_34_biasadd_readvariableop_resource:	9
'dense_35_matmul_readvariableop_resource:	6
(dense_35_biasadd_readvariableop_resource:
identityИвconv1d_6/BiasAdd/ReadVariableOpв+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpвconv1d_7/BiasAdd/ReadVariableOpв+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpвdense_35/BiasAdd/ReadVariableOpвdense_35/MatMul/ReadVariableOpв+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpв*lstm_34/lstm_cell_34/MatMul/ReadVariableOpв,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpвlstm_34/whileв+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpв*lstm_35/lstm_cell_35/MatMul/ReadVariableOpв,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpвlstm_35/whileЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_6/conv1d/ExpandDims/dim▒
conv1d_6/conv1d/ExpandDims
ExpandDimsinputs'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d_6/conv1d/ExpandDims╙
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim█
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_6/conv1d/ExpandDims_1┌
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv1d_6/conv1dн
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_6/conv1d/Squeezeз
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp░
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_6/BiasAddw
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_6/ReluЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_7/conv1d/ExpandDims/dim╞
conv1d_7/conv1d/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d_7/conv1d/ExpandDims╙
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim█
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_7/conv1d/ExpandDims_1┌
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv1d_7/conv1dн
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

¤        2
conv1d_7/conv1d/Squeezeз
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_7/BiasAdd/ReadVariableOp░
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:         @2
conv1d_7/ReluВ
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim╞
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @2
max_pooling1d_2/ExpandDims╬
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPoolм
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
2
max_pooling1d_2/Squeezen
lstm_34/ShapeShape max_pooling1d_2/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_34/ShapeД
lstm_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice/stackИ
lstm_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_34/strided_slice/stack_1И
lstm_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_34/strided_slice/stack_2Т
lstm_34/strided_sliceStridedSlicelstm_34/Shape:output:0$lstm_34/strided_slice/stack:output:0&lstm_34/strided_slice/stack_1:output:0&lstm_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_34/strided_slicel
lstm_34/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros/mul/yМ
lstm_34/zeros/mulMullstm_34/strided_slice:output:0lstm_34/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros/mulo
lstm_34/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_34/zeros/Less/yЗ
lstm_34/zeros/LessLesslstm_34/zeros/mul:z:0lstm_34/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros/Lessr
lstm_34/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros/packed/1г
lstm_34/zeros/packedPacklstm_34/strided_slice:output:0lstm_34/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_34/zeros/packedo
lstm_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/zeros/ConstХ
lstm_34/zerosFilllstm_34/zeros/packed:output:0lstm_34/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_34/zerosp
lstm_34/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros_1/mul/yТ
lstm_34/zeros_1/mulMullstm_34/strided_slice:output:0lstm_34/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros_1/muls
lstm_34/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_34/zeros_1/Less/yП
lstm_34/zeros_1/LessLesslstm_34/zeros_1/mul:z:0lstm_34/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_34/zeros_1/Lessv
lstm_34/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_34/zeros_1/packed/1й
lstm_34/zeros_1/packedPacklstm_34/strided_slice:output:0!lstm_34/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_34/zeros_1/packeds
lstm_34/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/zeros_1/ConstЭ
lstm_34/zeros_1Filllstm_34/zeros_1/packed:output:0lstm_34/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_34/zeros_1Е
lstm_34/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_34/transpose/permм
lstm_34/transpose	Transpose max_pooling1d_2/Squeeze:output:0lstm_34/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_34/transposeg
lstm_34/Shape_1Shapelstm_34/transpose:y:0*
T0*
_output_shapes
:2
lstm_34/Shape_1И
lstm_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice_1/stackМ
lstm_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_1/stack_1М
lstm_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_1/stack_2Ю
lstm_34/strided_slice_1StridedSlicelstm_34/Shape_1:output:0&lstm_34/strided_slice_1/stack:output:0(lstm_34/strided_slice_1/stack_1:output:0(lstm_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_34/strided_slice_1Х
#lstm_34/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_34/TensorArrayV2/element_shape╥
lstm_34/TensorArrayV2TensorListReserve,lstm_34/TensorArrayV2/element_shape:output:0 lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_34/TensorArrayV2╧
=lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2?
=lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_34/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_34/transpose:y:0Flstm_34/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_34/TensorArrayUnstack/TensorListFromTensorИ
lstm_34/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_34/strided_slice_2/stackМ
lstm_34/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_2/stack_1М
lstm_34/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_2/stack_2м
lstm_34/strided_slice_2StridedSlicelstm_34/transpose:y:0&lstm_34/strided_slice_2/stack:output:0(lstm_34/strided_slice_2/stack_1:output:0(lstm_34/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_34/strided_slice_2╠
*lstm_34/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3lstm_34_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02,
*lstm_34/lstm_cell_34/MatMul/ReadVariableOp╠
lstm_34/lstm_cell_34/MatMulMatMul lstm_34/strided_slice_2:output:02lstm_34/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/MatMul╥
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02.
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp╚
lstm_34/lstm_cell_34/MatMul_1MatMullstm_34/zeros:output:04lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/MatMul_1┐
lstm_34/lstm_cell_34/addAddV2%lstm_34/lstm_cell_34/MatMul:product:0'lstm_34/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/add╦
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp╠
lstm_34/lstm_cell_34/BiasAddBiasAddlstm_34/lstm_cell_34/add:z:03lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_34/lstm_cell_34/BiasAddО
$lstm_34/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_34/lstm_cell_34/split/split_dimУ
lstm_34/lstm_cell_34/splitSplit-lstm_34/lstm_cell_34/split/split_dim:output:0%lstm_34/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_34/lstm_cell_34/splitЮ
lstm_34/lstm_cell_34/SigmoidSigmoid#lstm_34/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Sigmoidв
lstm_34/lstm_cell_34/Sigmoid_1Sigmoid#lstm_34/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2 
lstm_34/lstm_cell_34/Sigmoid_1л
lstm_34/lstm_cell_34/mulMul"lstm_34/lstm_cell_34/Sigmoid_1:y:0lstm_34/zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mulХ
lstm_34/lstm_cell_34/ReluRelu#lstm_34/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Relu╝
lstm_34/lstm_cell_34/mul_1Mul lstm_34/lstm_cell_34/Sigmoid:y:0'lstm_34/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mul_1▒
lstm_34/lstm_cell_34/add_1AddV2lstm_34/lstm_cell_34/mul:z:0lstm_34/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/add_1в
lstm_34/lstm_cell_34/Sigmoid_2Sigmoid#lstm_34/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2 
lstm_34/lstm_cell_34/Sigmoid_2Ф
lstm_34/lstm_cell_34/Relu_1Relulstm_34/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/Relu_1└
lstm_34/lstm_cell_34/mul_2Mul"lstm_34/lstm_cell_34/Sigmoid_2:y:0)lstm_34/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_34/lstm_cell_34/mul_2Я
%lstm_34/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2'
%lstm_34/TensorArrayV2_1/element_shape╪
lstm_34/TensorArrayV2_1TensorListReserve.lstm_34/TensorArrayV2_1/element_shape:output:0 lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_34/TensorArrayV2_1^
lstm_34/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_34/timeП
 lstm_34/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_34/while/maximum_iterationsz
lstm_34/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_34/while/loop_counterГ
lstm_34/whileWhile#lstm_34/while/loop_counter:output:0)lstm_34/while/maximum_iterations:output:0lstm_34/time:output:0 lstm_34/TensorArrayV2_1:handle:0lstm_34/zeros:output:0lstm_34/zeros_1:output:0 lstm_34/strided_slice_1:output:0?lstm_34/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_34_lstm_cell_34_matmul_readvariableop_resource5lstm_34_lstm_cell_34_matmul_1_readvariableop_resource4lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_34_while_body_321830*%
condR
lstm_34_while_cond_321829*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
lstm_34/while┼
8lstm_34/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2:
8lstm_34/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_34/TensorArrayV2Stack/TensorListStackTensorListStacklstm_34/while:output:3Alstm_34/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02,
*lstm_34/TensorArrayV2Stack/TensorListStackС
lstm_34/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_34/strided_slice_3/stackМ
lstm_34/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_34/strided_slice_3/stack_1М
lstm_34/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_34/strided_slice_3/stack_2╩
lstm_34/strided_slice_3StridedSlice3lstm_34/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_34/strided_slice_3/stack:output:0(lstm_34/strided_slice_3/stack_1:output:0(lstm_34/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_34/strided_slice_3Й
lstm_34/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_34/transpose_1/perm┼
lstm_34/transpose_1	Transpose3lstm_34/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_34/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_34/transpose_1v
lstm_34/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_34/runtimeЕ
dropout_18/IdentityIdentitylstm_34/transpose_1:y:0*
T0*+
_output_shapes
:         	2
dropout_18/Identityj
lstm_35/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:2
lstm_35/ShapeД
lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice/stackИ
lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_35/strided_slice/stack_1И
lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_35/strided_slice/stack_2Т
lstm_35/strided_sliceStridedSlicelstm_35/Shape:output:0$lstm_35/strided_slice/stack:output:0&lstm_35/strided_slice/stack_1:output:0&lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_35/strided_slicel
lstm_35/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros/mul/yМ
lstm_35/zeros/mulMullstm_35/strided_slice:output:0lstm_35/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros/mulo
lstm_35/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_35/zeros/Less/yЗ
lstm_35/zeros/LessLesslstm_35/zeros/mul:z:0lstm_35/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros/Lessr
lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros/packed/1г
lstm_35/zeros/packedPacklstm_35/strided_slice:output:0lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_35/zeros/packedo
lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/zeros/ConstХ
lstm_35/zerosFilllstm_35/zeros/packed:output:0lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_35/zerosp
lstm_35/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros_1/mul/yТ
lstm_35/zeros_1/mulMullstm_35/strided_slice:output:0lstm_35/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros_1/muls
lstm_35/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_35/zeros_1/Less/yП
lstm_35/zeros_1/LessLesslstm_35/zeros_1/mul:z:0lstm_35/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_35/zeros_1/Lessv
lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2
lstm_35/zeros_1/packed/1й
lstm_35/zeros_1/packedPacklstm_35/strided_slice:output:0!lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_35/zeros_1/packeds
lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/zeros_1/ConstЭ
lstm_35/zeros_1Filllstm_35/zeros_1/packed:output:0lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
lstm_35/zeros_1Е
lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_35/transpose/permи
lstm_35/transpose	Transposedropout_18/Identity:output:0lstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_35/transposeg
lstm_35/Shape_1Shapelstm_35/transpose:y:0*
T0*
_output_shapes
:2
lstm_35/Shape_1И
lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice_1/stackМ
lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_1/stack_1М
lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_1/stack_2Ю
lstm_35/strided_slice_1StridedSlicelstm_35/Shape_1:output:0&lstm_35/strided_slice_1/stack:output:0(lstm_35/strided_slice_1/stack_1:output:0(lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_35/strided_slice_1Х
#lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_35/TensorArrayV2/element_shape╥
lstm_35/TensorArrayV2TensorListReserve,lstm_35/TensorArrayV2/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_35/TensorArrayV2╧
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2?
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_35/transpose:y:0Flstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_35/TensorArrayUnstack/TensorListFromTensorИ
lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_35/strided_slice_2/stackМ
lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_2/stack_1М
lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_2/stack_2м
lstm_35/strided_slice_2StridedSlicelstm_35/transpose:y:0&lstm_35/strided_slice_2/stack:output:0(lstm_35/strided_slice_2/stack_1:output:0(lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_35/strided_slice_2╠
*lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02,
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp╠
lstm_35/lstm_cell_35/MatMulMatMul lstm_35/strided_slice_2:output:02lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/MatMul╥
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02.
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp╚
lstm_35/lstm_cell_35/MatMul_1MatMullstm_35/zeros:output:04lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/MatMul_1┐
lstm_35/lstm_cell_35/addAddV2%lstm_35/lstm_cell_35/MatMul:product:0'lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/add╦
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp╠
lstm_35/lstm_cell_35/BiasAddBiasAddlstm_35/lstm_cell_35/add:z:03lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_35/lstm_cell_35/BiasAddО
$lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_35/lstm_cell_35/split/split_dimУ
lstm_35/lstm_cell_35/splitSplit-lstm_35/lstm_cell_35/split/split_dim:output:0%lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_35/lstm_cell_35/splitЮ
lstm_35/lstm_cell_35/SigmoidSigmoid#lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Sigmoidв
lstm_35/lstm_cell_35/Sigmoid_1Sigmoid#lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2 
lstm_35/lstm_cell_35/Sigmoid_1л
lstm_35/lstm_cell_35/mulMul"lstm_35/lstm_cell_35/Sigmoid_1:y:0lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mulХ
lstm_35/lstm_cell_35/ReluRelu#lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Relu╝
lstm_35/lstm_cell_35/mul_1Mul lstm_35/lstm_cell_35/Sigmoid:y:0'lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mul_1▒
lstm_35/lstm_cell_35/add_1AddV2lstm_35/lstm_cell_35/mul:z:0lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/add_1в
lstm_35/lstm_cell_35/Sigmoid_2Sigmoid#lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2 
lstm_35/lstm_cell_35/Sigmoid_2Ф
lstm_35/lstm_cell_35/Relu_1Relulstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/Relu_1└
lstm_35/lstm_cell_35/mul_2Mul"lstm_35/lstm_cell_35/Sigmoid_2:y:0)lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_35/lstm_cell_35/mul_2Я
%lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2'
%lstm_35/TensorArrayV2_1/element_shape╪
lstm_35/TensorArrayV2_1TensorListReserve.lstm_35/TensorArrayV2_1/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_35/TensorArrayV2_1^
lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_35/timeП
 lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_35/while/maximum_iterationsz
lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_35/while/loop_counterГ
lstm_35/whileWhile#lstm_35/while/loop_counter:output:0)lstm_35/while/maximum_iterations:output:0lstm_35/time:output:0 lstm_35/TensorArrayV2_1:handle:0lstm_35/zeros:output:0lstm_35/zeros_1:output:0 lstm_35/strided_slice_1:output:0?lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_35_lstm_cell_35_matmul_readvariableop_resource5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_35_while_body_321978*%
condR
lstm_35_while_cond_321977*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
lstm_35/while┼
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2:
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_35/TensorArrayV2Stack/TensorListStackTensorListStacklstm_35/while:output:3Alstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02,
*lstm_35/TensorArrayV2Stack/TensorListStackС
lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_35/strided_slice_3/stackМ
lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_35/strided_slice_3/stack_1М
lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_35/strided_slice_3/stack_2╩
lstm_35/strided_slice_3StridedSlice3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_35/strided_slice_3/stack:output:0(lstm_35/strided_slice_3/stack_1:output:0(lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2
lstm_35/strided_slice_3Й
lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_35/transpose_1/perm┼
lstm_35/transpose_1	Transpose3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2
lstm_35/transpose_1v
lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_35/runtimeК
dropout_19/IdentityIdentity lstm_35/strided_slice_3:output:0*
T0*'
_output_shapes
:         	2
dropout_19/Identityи
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02 
dense_34/MatMul/ReadVariableOpд
dense_34/MatMulMatMuldropout_19/Identity:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
dense_34/MatMulз
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_34/BiasAdd/ReadVariableOpе
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
dense_34/Reluи
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02 
dense_35/MatMul/ReadVariableOpг
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_35/MatMulз
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOpе
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_35/BiasAddm
reshape_17/ShapeShapedense_35/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_17/ShapeК
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stackО
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1О
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2д
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2╫
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shapeз
reshape_17/ReshapeReshapedense_35/BiasAdd:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_17/Reshapez
IdentityIdentityreshape_17/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityи
NoOpNoOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp,^lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp+^lstm_34/lstm_cell_34/MatMul/ReadVariableOp-^lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp^lstm_34/while,^lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+^lstm_35/lstm_cell_35/MatMul/ReadVariableOp-^lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp^lstm_35/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2Z
+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp+lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp2X
*lstm_34/lstm_cell_34/MatMul/ReadVariableOp*lstm_34/lstm_cell_34/MatMul/ReadVariableOp2\
,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp,lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp2
lstm_34/whilelstm_34/while2Z
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2X
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp*lstm_35/lstm_cell_35/MatMul/ReadVariableOp2\
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2
lstm_35/whilelstm_35/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╨
G
+__inference_dropout_18_layer_call_fn_323178

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3207132
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Г
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_323188

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         	2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
▄[
Ш
C__inference_lstm_35_layer_call_and_return_conditional_losses_321173

inputs=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
:         	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_321089*
condR
while_cond_321088*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
│2
╓
I__inference_sequential_12_layer_call_and_return_conditional_losses_321480

inputs%
conv1d_6_321441: 
conv1d_6_321443: %
conv1d_7_321446: @
conv1d_7_321448:@ 
lstm_34_321452:@$ 
lstm_34_321454:	$
lstm_34_321456:$ 
lstm_35_321460:	$ 
lstm_35_321462:	$
lstm_35_321464:$!
dense_34_321468:		
dense_34_321470:	!
dense_35_321473:	
dense_35_321475:
identityИв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCallвlstm_34/StatefulPartitionedCallвlstm_35/StatefulPartitionedCallШ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_6_321441conv1d_6_321443*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3205132"
 conv1d_6/StatefulPartitionedCall╗
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0conv1d_7_321446conv1d_7_321448*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3205352"
 conv1d_7/StatefulPartitionedCallР
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_3205482!
max_pooling1d_2/PartitionedCall╟
lstm_34/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0lstm_34_321452lstm_34_321454lstm_34_321456*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3213692!
lstm_34/StatefulPartitionedCallШ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_3212022$
"dropout_18/StatefulPartitionedCall╞
lstm_35/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_35_321460lstm_35_321462lstm_35_321464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3211732!
lstm_35/StatefulPartitionedCall╣
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3210062$
"dropout_19/StatefulPartitionedCall╣
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_34_321468dense_34_321470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3208912"
 dense_34/StatefulPartitionedCall╖
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_321473dense_35_321475*
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
GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_3209072"
 dense_35/StatefulPartitionedCallБ
reshape_17/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *O
fJRH
F__inference_reshape_17_layer_call_and_return_conditional_losses_3209262
reshape_17/PartitionedCallВ
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityш
NoOpNoOp!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_34/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_34/StatefulPartitionedCalllstm_34/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╚
G
+__inference_reshape_17_layer_call_fn_323919

inputs
identity╚
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
GPU 2J 8В *O
fJRH
F__inference_reshape_17_layer_call_and_return_conditional_losses_3209262
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
╞

у
lstm_34_while_cond_321829,
(lstm_34_while_lstm_34_while_loop_counter2
.lstm_34_while_lstm_34_while_maximum_iterations
lstm_34_while_placeholder
lstm_34_while_placeholder_1
lstm_34_while_placeholder_2
lstm_34_while_placeholder_3.
*lstm_34_while_less_lstm_34_strided_slice_1D
@lstm_34_while_lstm_34_while_cond_321829___redundant_placeholder0D
@lstm_34_while_lstm_34_while_cond_321829___redundant_placeholder1D
@lstm_34_while_lstm_34_while_cond_321829___redundant_placeholder2D
@lstm_34_while_lstm_34_while_cond_321829___redundant_placeholder3
lstm_34_while_identity
Ш
lstm_34/while/LessLesslstm_34_while_placeholder*lstm_34_while_less_lstm_34_strided_slice_1*
T0*
_output_shapes
: 2
lstm_34/while/Lessu
lstm_34/while/IdentityIdentitylstm_34/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_34/while/Identity"9
lstm_34_while_identitylstm_34/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
ЬF
В
C__inference_lstm_34_layer_call_and_return_conditional_losses_319598

inputs%
lstm_cell_34_319516:@$%
lstm_cell_34_319518:	$!
lstm_cell_34_319520:$
identityИв$lstm_cell_34/StatefulPartitionedCallвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2Э
$lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_34_319516lstm_cell_34_319518lstm_cell_34_319520*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3194512&
$lstm_cell_34/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counter└
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_34_319516lstm_cell_34_319518lstm_cell_34_319520*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319529*
condR
while_cond_319528*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
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
 :                  	2

Identity}
NoOpNoOp%^lstm_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2L
$lstm_cell_34/StatefulPartitionedCall$lstm_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
└J
╩

lstm_34_while_body_322180,
(lstm_34_while_lstm_34_while_loop_counter2
.lstm_34_while_lstm_34_while_maximum_iterations
lstm_34_while_placeholder
lstm_34_while_placeholder_1
lstm_34_while_placeholder_2
lstm_34_while_placeholder_3+
'lstm_34_while_lstm_34_strided_slice_1_0g
clstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0:@$O
=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$J
<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0:$
lstm_34_while_identity
lstm_34_while_identity_1
lstm_34_while_identity_2
lstm_34_while_identity_3
lstm_34_while_identity_4
lstm_34_while_identity_5)
%lstm_34_while_lstm_34_strided_slice_1e
alstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensorK
9lstm_34_while_lstm_cell_34_matmul_readvariableop_resource:@$M
;lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource:	$H
:lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource:$Ив1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpв0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpв2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp╙
?lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2A
?lstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_34/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0lstm_34_while_placeholderHlstm_34/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype023
1lstm_34/while/TensorArrayV2Read/TensorListGetItemр
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype022
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOpЎ
!lstm_34/while/lstm_cell_34/MatMulMatMul8lstm_34/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2#
!lstm_34/while/lstm_cell_34/MatMulц
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype024
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp▀
#lstm_34/while/lstm_cell_34/MatMul_1MatMullstm_34_while_placeholder_2:lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2%
#lstm_34/while/lstm_cell_34/MatMul_1╫
lstm_34/while/lstm_cell_34/addAddV2+lstm_34/while/lstm_cell_34/MatMul:product:0-lstm_34/while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2 
lstm_34/while/lstm_cell_34/add▀
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype023
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOpф
"lstm_34/while/lstm_cell_34/BiasAddBiasAdd"lstm_34/while/lstm_cell_34/add:z:09lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2$
"lstm_34/while/lstm_cell_34/BiasAddЪ
*lstm_34/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_34/while/lstm_cell_34/split/split_dimл
 lstm_34/while/lstm_cell_34/splitSplit3lstm_34/while/lstm_cell_34/split/split_dim:output:0+lstm_34/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2"
 lstm_34/while/lstm_cell_34/split░
"lstm_34/while/lstm_cell_34/SigmoidSigmoid)lstm_34/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2$
"lstm_34/while/lstm_cell_34/Sigmoid┤
$lstm_34/while/lstm_cell_34/Sigmoid_1Sigmoid)lstm_34/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2&
$lstm_34/while/lstm_cell_34/Sigmoid_1└
lstm_34/while/lstm_cell_34/mulMul(lstm_34/while/lstm_cell_34/Sigmoid_1:y:0lstm_34_while_placeholder_3*
T0*'
_output_shapes
:         	2 
lstm_34/while/lstm_cell_34/mulз
lstm_34/while/lstm_cell_34/ReluRelu)lstm_34/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2!
lstm_34/while/lstm_cell_34/Relu╘
 lstm_34/while/lstm_cell_34/mul_1Mul&lstm_34/while/lstm_cell_34/Sigmoid:y:0-lstm_34/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/mul_1╔
 lstm_34/while/lstm_cell_34/add_1AddV2"lstm_34/while/lstm_cell_34/mul:z:0$lstm_34/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/add_1┤
$lstm_34/while/lstm_cell_34/Sigmoid_2Sigmoid)lstm_34/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2&
$lstm_34/while/lstm_cell_34/Sigmoid_2ж
!lstm_34/while/lstm_cell_34/Relu_1Relu$lstm_34/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2#
!lstm_34/while/lstm_cell_34/Relu_1╪
 lstm_34/while/lstm_cell_34/mul_2Mul(lstm_34/while/lstm_cell_34/Sigmoid_2:y:0/lstm_34/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_34/while/lstm_cell_34/mul_2И
2lstm_34/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_34_while_placeholder_1lstm_34_while_placeholder$lstm_34/while/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_34/while/TensorArrayV2Write/TensorListSetIteml
lstm_34/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_34/while/add/yЙ
lstm_34/while/addAddV2lstm_34_while_placeholderlstm_34/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_34/while/addp
lstm_34/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_34/while/add_1/yЮ
lstm_34/while/add_1AddV2(lstm_34_while_lstm_34_while_loop_counterlstm_34/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_34/while/add_1Л
lstm_34/while/IdentityIdentitylstm_34/while/add_1:z:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identityж
lstm_34/while/Identity_1Identity.lstm_34_while_lstm_34_while_maximum_iterations^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_1Н
lstm_34/while/Identity_2Identitylstm_34/while/add:z:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_2║
lstm_34/while/Identity_3IdentityBlstm_34/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_34/while/NoOp*
T0*
_output_shapes
: 2
lstm_34/while/Identity_3н
lstm_34/while/Identity_4Identity$lstm_34/while/lstm_cell_34/mul_2:z:0^lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_34/while/Identity_4н
lstm_34/while/Identity_5Identity$lstm_34/while/lstm_cell_34/add_1:z:0^lstm_34/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_34/while/Identity_5Ж
lstm_34/while/NoOpNoOp2^lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp1^lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp3^lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_34/while/NoOp"9
lstm_34_while_identitylstm_34/while/Identity:output:0"=
lstm_34_while_identity_1!lstm_34/while/Identity_1:output:0"=
lstm_34_while_identity_2!lstm_34/while/Identity_2:output:0"=
lstm_34_while_identity_3!lstm_34/while/Identity_3:output:0"=
lstm_34_while_identity_4!lstm_34/while/Identity_4:output:0"=
lstm_34_while_identity_5!lstm_34/while/Identity_5:output:0"P
%lstm_34_while_lstm_34_strided_slice_1'lstm_34_while_lstm_34_strided_slice_1_0"z
:lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource<lstm_34_while_lstm_cell_34_biasadd_readvariableop_resource_0"|
;lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource=lstm_34_while_lstm_cell_34_matmul_1_readvariableop_resource_0"x
9lstm_34_while_lstm_cell_34_matmul_readvariableop_resource;lstm_34_while_lstm_cell_34_matmul_readvariableop_resource_0"╚
alstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensorclstm_34_while_tensorarrayv2read_tensorlistgetitem_lstm_34_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2f
1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp1lstm_34/while/lstm_cell_34/BiasAdd/ReadVariableOp2d
0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp0lstm_34/while/lstm_cell_34/MatMul/ReadVariableOp2h
2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp2lstm_34/while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_322786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322786___redundant_placeholder04
0while_while_cond_322786___redundant_placeholder14
0while_while_cond_322786___redundant_placeholder24
0while_while_cond_322786___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Г
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_320713

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         	2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
ы
√
'sequential_12_lstm_35_while_cond_319094H
Dsequential_12_lstm_35_while_sequential_12_lstm_35_while_loop_counterN
Jsequential_12_lstm_35_while_sequential_12_lstm_35_while_maximum_iterations+
'sequential_12_lstm_35_while_placeholder-
)sequential_12_lstm_35_while_placeholder_1-
)sequential_12_lstm_35_while_placeholder_2-
)sequential_12_lstm_35_while_placeholder_3J
Fsequential_12_lstm_35_while_less_sequential_12_lstm_35_strided_slice_1`
\sequential_12_lstm_35_while_sequential_12_lstm_35_while_cond_319094___redundant_placeholder0`
\sequential_12_lstm_35_while_sequential_12_lstm_35_while_cond_319094___redundant_placeholder1`
\sequential_12_lstm_35_while_sequential_12_lstm_35_while_cond_319094___redundant_placeholder2`
\sequential_12_lstm_35_while_sequential_12_lstm_35_while_cond_319094___redundant_placeholder3(
$sequential_12_lstm_35_while_identity
▐
 sequential_12/lstm_35/while/LessLess'sequential_12_lstm_35_while_placeholderFsequential_12_lstm_35_while_less_sequential_12_lstm_35_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_35/while/LessЯ
$sequential_12/lstm_35/while/IdentityIdentity$sequential_12/lstm_35/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_35/while/Identity"U
$sequential_12_lstm_35_while_identity-sequential_12/lstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
│
є
-__inference_lstm_cell_35_layer_call_fn_324047

inputs
states_0
states_1
unknown:	$
	unknown_0:	$
	unknown_1:$
identity

identity_1

identity_2ИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_3199352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         	2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
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
Л?
╩
while_body_321285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
Т\
Ъ
C__inference_lstm_35_layer_call_and_return_conditional_losses_323546
inputs_0=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileF
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
 :                  	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323462*
condR
while_cond_323461*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
Л?
╩
while_body_322636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_34_matmul_readvariableop_resource_0:@$G
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_34_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_34_matmul_readvariableop_resource:@$E
3while_lstm_cell_34_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_34_biasadd_readvariableop_resource:$Ив)while/lstm_cell_34/BiasAdd/ReadVariableOpв(while/lstm_cell_34/MatMul/ReadVariableOpв*while/lstm_cell_34/MatMul_1/ReadVariableOp├
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
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:@$*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp╓
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul╬
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp┐
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/MatMul_1╖
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/add╟
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp─
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_34/BiasAddК
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dimЛ
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_34/splitШ
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/SigmoidЬ
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_1а
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mulП
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu┤
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_1й
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/add_1Ь
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Sigmoid_2О
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/Relu_1╕
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_34/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_323763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323763___redundant_placeholder04
0while_while_cond_323763___redundant_placeholder14
0while_while_cond_323763___redundant_placeholder24
0while_while_cond_323763___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
ж

ї
D__inference_dense_35_layer_call_and_return_conditional_losses_320907

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
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
_construction_contextkEagerRuntime**
_input_shapes
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
ё
Г
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_323998

inputs
states_0
states_10
matmul_readvariableop_resource:@$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
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
▄[
Ш
C__inference_lstm_35_layer_call_and_return_conditional_losses_320865

inputs=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
:         	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320781*
condR
while_cond_320780*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
ЬF
В
C__inference_lstm_34_layer_call_and_return_conditional_losses_319388

inputs%
lstm_cell_34_319306:@$%
lstm_cell_34_319308:	$!
lstm_cell_34_319310:$
identityИв$lstm_cell_34/StatefulPartitionedCallвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
strided_slice_2Э
$lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_34_319306lstm_cell_34_319308lstm_cell_34_319310*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3193052&
$lstm_cell_34/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counter└
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_34_319306lstm_cell_34_319308lstm_cell_34_319310*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319319*
condR
while_cond_319318*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
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
 :                  	2

Identity}
NoOpNoOp%^lstm_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2L
$lstm_cell_34/StatefulPartitionedCall$lstm_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
л
▓
(__inference_lstm_34_layer_call_fn_322569

inputs
unknown:@$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3213692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         	2

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
▄[
Ш
C__inference_lstm_35_layer_call_and_return_conditional_losses_323848

inputs=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileD
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
:         	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323764*
condR
while_cond_323763*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
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
:         	*
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
:         	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
╞

у
lstm_34_while_cond_322179,
(lstm_34_while_lstm_34_while_loop_counter2
.lstm_34_while_lstm_34_while_maximum_iterations
lstm_34_while_placeholder
lstm_34_while_placeholder_1
lstm_34_while_placeholder_2
lstm_34_while_placeholder_3.
*lstm_34_while_less_lstm_34_strided_slice_1D
@lstm_34_while_lstm_34_while_cond_322179___redundant_placeholder0D
@lstm_34_while_lstm_34_while_cond_322179___redundant_placeholder1D
@lstm_34_while_lstm_34_while_cond_322179___redundant_placeholder2D
@lstm_34_while_lstm_34_while_cond_322179___redundant_placeholder3
lstm_34_while_identity
Ш
lstm_34/while/LessLesslstm_34_while_placeholder*lstm_34_while_less_lstm_34_strided_slice_1*
T0*
_output_shapes
: 2
lstm_34/while/Lessu
lstm_34/while/IdentityIdentitylstm_34/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_34/while/Identity"9
lstm_34_while_identitylstm_34/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
╥%
▌
while_body_319319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_34_319343_0:@$-
while_lstm_cell_34_319345_0:	$)
while_lstm_cell_34_319347_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_34_319343:@$+
while_lstm_cell_34_319345:	$'
while_lstm_cell_34_319347:$Ив*while/lstm_cell_34/StatefulPartitionedCall├
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
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_34_319343_0while_lstm_cell_34_319345_0while_lstm_cell_34_319347_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3193052,
*while/lstm_cell_34/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_34/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_34/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_34/StatefulPartitionedCall*"
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
while_lstm_cell_34_319343while_lstm_cell_34_319343_0"8
while_lstm_cell_34_319345while_lstm_cell_34_319345_0"8
while_lstm_cell_34_319347while_lstm_cell_34_319347_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2X
*while/lstm_cell_34/StatefulPartitionedCall*while/lstm_cell_34/StatefulPartitionedCall: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
▄╓
Ц 
"__inference__traced_restore_324455
file_prefix6
 assignvariableop_conv1d_6_kernel: .
 assignvariableop_1_conv1d_6_bias: 8
"assignvariableop_2_conv1d_7_kernel: @.
 assignvariableop_3_conv1d_7_bias:@4
"assignvariableop_4_dense_34_kernel:		.
 assignvariableop_5_dense_34_bias:	4
"assignvariableop_6_dense_35_kernel:	.
 assignvariableop_7_dense_35_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: A
/assignvariableop_13_lstm_34_lstm_cell_34_kernel:@$K
9assignvariableop_14_lstm_34_lstm_cell_34_recurrent_kernel:	$;
-assignvariableop_15_lstm_34_lstm_cell_34_bias:$A
/assignvariableop_16_lstm_35_lstm_cell_35_kernel:	$K
9assignvariableop_17_lstm_35_lstm_cell_35_recurrent_kernel:	$;
-assignvariableop_18_lstm_35_lstm_cell_35_bias:$#
assignvariableop_19_total: #
assignvariableop_20_count: @
*assignvariableop_21_adam_conv1d_6_kernel_m: 6
(assignvariableop_22_adam_conv1d_6_bias_m: @
*assignvariableop_23_adam_conv1d_7_kernel_m: @6
(assignvariableop_24_adam_conv1d_7_bias_m:@<
*assignvariableop_25_adam_dense_34_kernel_m:		6
(assignvariableop_26_adam_dense_34_bias_m:	<
*assignvariableop_27_adam_dense_35_kernel_m:	6
(assignvariableop_28_adam_dense_35_bias_m:H
6assignvariableop_29_adam_lstm_34_lstm_cell_34_kernel_m:@$R
@assignvariableop_30_adam_lstm_34_lstm_cell_34_recurrent_kernel_m:	$B
4assignvariableop_31_adam_lstm_34_lstm_cell_34_bias_m:$H
6assignvariableop_32_adam_lstm_35_lstm_cell_35_kernel_m:	$R
@assignvariableop_33_adam_lstm_35_lstm_cell_35_recurrent_kernel_m:	$B
4assignvariableop_34_adam_lstm_35_lstm_cell_35_bias_m:$@
*assignvariableop_35_adam_conv1d_6_kernel_v: 6
(assignvariableop_36_adam_conv1d_6_bias_v: @
*assignvariableop_37_adam_conv1d_7_kernel_v: @6
(assignvariableop_38_adam_conv1d_7_bias_v:@<
*assignvariableop_39_adam_dense_34_kernel_v:		6
(assignvariableop_40_adam_dense_34_bias_v:	<
*assignvariableop_41_adam_dense_35_kernel_v:	6
(assignvariableop_42_adam_dense_35_bias_v:H
6assignvariableop_43_adam_lstm_34_lstm_cell_34_kernel_v:@$R
@assignvariableop_44_adam_lstm_34_lstm_cell_34_recurrent_kernel_v:	$B
4assignvariableop_45_adam_lstm_34_lstm_cell_34_bias_v:$H
6assignvariableop_46_adam_lstm_35_lstm_cell_35_kernel_v:	$R
@assignvariableop_47_adam_lstm_35_lstm_cell_35_recurrent_kernel_v:	$B
4assignvariableop_48_adam_lstm_35_lstm_cell_35_bias_v:$
identity_50ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9┤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*└
value╢B│2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesи
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▐
_output_shapes╦
╚::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv1d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4з
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_34_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5е
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_34_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6з
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_35_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7е
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_35_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8б
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10з
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ж
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12о
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╖
AssignVariableOp_13AssignVariableOp/assignvariableop_13_lstm_34_lstm_cell_34_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14┴
AssignVariableOp_14AssignVariableOp9assignvariableop_14_lstm_34_lstm_cell_34_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╡
AssignVariableOp_15AssignVariableOp-assignvariableop_15_lstm_34_lstm_cell_34_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╖
AssignVariableOp_16AssignVariableOp/assignvariableop_16_lstm_35_lstm_cell_35_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┴
AssignVariableOp_17AssignVariableOp9assignvariableop_17_lstm_35_lstm_cell_35_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╡
AssignVariableOp_18AssignVariableOp-assignvariableop_18_lstm_35_lstm_cell_35_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19б
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20б
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21▓
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_6_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22░
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_6_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24░
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_7_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_34_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_34_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_35_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_35_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╛
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_34_lstm_cell_34_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╚
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_34_lstm_cell_34_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╝
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_34_lstm_cell_34_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╛
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_35_lstm_cell_35_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╚
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_35_lstm_cell_35_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╝
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_35_lstm_cell_35_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_6_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_6_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_7_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_7_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_34_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_34_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41▓
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_35_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42░
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_35_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╛
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_lstm_34_lstm_cell_34_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╚
AssignVariableOp_44AssignVariableOp@assignvariableop_44_adam_lstm_34_lstm_cell_34_recurrent_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╝
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_lstm_34_lstm_cell_34_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╛
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_lstm_35_lstm_cell_35_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╚
AssignVariableOp_47AssignVariableOp@assignvariableop_47_adam_lstm_35_lstm_cell_35_recurrent_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48╝
AssignVariableOp_48AssignVariableOp4assignvariableop_48_adam_lstm_35_lstm_cell_35_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpФ	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49f
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_50№
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
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
щ
Б
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_319451

inputs

states
states_10
matmul_readvariableop_resource:@$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_namestates:OK
'
_output_shapes
:         	
 
_user_specified_namestates
╒
├
while_cond_321284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_321284___redundant_placeholder04
0while_while_cond_321284___redundant_placeholder14
0while_while_cond_321284___redundant_placeholder24
0while_while_cond_321284___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
ё
Ц
)__inference_dense_35_layer_call_fn_323904

inputs
unknown:	
	unknown_0:
identityИвStatefulPartitionedCallЇ
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
GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_3209072
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
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
є
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_320878

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         	2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
С
g
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_319214

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
г
▓
(__inference_lstm_35_layer_call_fn_323233

inputs
unknown:	$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_3208652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Л?
╩
while_body_320781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
│
є
-__inference_lstm_cell_34_layer_call_fn_323966

inputs
states_0
states_1
unknown:@$
	unknown_0:	$
	unknown_1:$
identity

identity_1

identity_2ИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_3194512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         	2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         	2

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
?:         @:         	:         	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
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
Т\
Ъ
C__inference_lstm_35_layer_call_and_return_conditional_losses_323395
inputs_0=
+lstm_cell_35_matmul_readvariableop_resource:	$?
-lstm_cell_35_matmul_1_readvariableop_resource:	$:
,lstm_cell_35_biasadd_readvariableop_resource:$
identityИв#lstm_cell_35/BiasAdd/ReadVariableOpв"lstm_cell_35/MatMul/ReadVariableOpв$lstm_cell_35/MatMul_1/ReadVariableOpвwhileF
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
value	B :	2
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
value	B :	2
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
:         	2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2
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
value	B :	2
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
:         	2	
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
 :                  	2
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
valueB"    	   27
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
:         	*
shrink_axis_mask2
strided_slice_2┤
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOpм
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul║
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOpи
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/MatMul_1Я
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/add│
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOpм
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dimє
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
lstm_cell_35/splitЖ
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/SigmoidК
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_1Л
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
lstm_cell_35/ReluЬ
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_1С
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/add_1К
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/Relu_1а
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
lstm_cell_35/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2
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
while/loop_counterЛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323311*
condR
while_cond_323310*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  	*
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
:         	*
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
 :                  	2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         	2

Identity╚
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
Ьу
▄
!__inference__wrapped_model_319202
conv1d_6_inputX
Bsequential_12_conv1d_6_conv1d_expanddims_1_readvariableop_resource: D
6sequential_12_conv1d_6_biasadd_readvariableop_resource: X
Bsequential_12_conv1d_7_conv1d_expanddims_1_readvariableop_resource: @D
6sequential_12_conv1d_7_biasadd_readvariableop_resource:@S
Asequential_12_lstm_34_lstm_cell_34_matmul_readvariableop_resource:@$U
Csequential_12_lstm_34_lstm_cell_34_matmul_1_readvariableop_resource:	$P
Bsequential_12_lstm_34_lstm_cell_34_biasadd_readvariableop_resource:$S
Asequential_12_lstm_35_lstm_cell_35_matmul_readvariableop_resource:	$U
Csequential_12_lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	$P
Bsequential_12_lstm_35_lstm_cell_35_biasadd_readvariableop_resource:$G
5sequential_12_dense_34_matmul_readvariableop_resource:		D
6sequential_12_dense_34_biasadd_readvariableop_resource:	G
5sequential_12_dense_35_matmul_readvariableop_resource:	D
6sequential_12_dense_35_biasadd_readvariableop_resource:
identityИв-sequential_12/conv1d_6/BiasAdd/ReadVariableOpв9sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpв-sequential_12/conv1d_7/BiasAdd/ReadVariableOpв9sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpв-sequential_12/dense_34/BiasAdd/ReadVariableOpв,sequential_12/dense_34/MatMul/ReadVariableOpв-sequential_12/dense_35/BiasAdd/ReadVariableOpв,sequential_12/dense_35/MatMul/ReadVariableOpв9sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpв8sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOpв:sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpвsequential_12/lstm_34/whileв9sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpв8sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOpв:sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpвsequential_12/lstm_35/whileз
,sequential_12/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2.
,sequential_12/conv1d_6/conv1d/ExpandDims/dimу
(sequential_12/conv1d_6/conv1d/ExpandDims
ExpandDimsconv1d_6_input5sequential_12/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2*
(sequential_12/conv1d_6/conv1d/ExpandDims¤
9sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBsequential_12_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02;
9sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpв
.sequential_12/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/conv1d_6/conv1d/ExpandDims_1/dimУ
*sequential_12/conv1d_6/conv1d/ExpandDims_1
ExpandDimsAsequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:07sequential_12/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2,
*sequential_12/conv1d_6/conv1d/ExpandDims_1Т
sequential_12/conv1d_6/conv1dConv2D1sequential_12/conv1d_6/conv1d/ExpandDims:output:03sequential_12/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_12/conv1d_6/conv1d╫
%sequential_12/conv1d_6/conv1d/SqueezeSqueeze&sequential_12/conv1d_6/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2'
%sequential_12/conv1d_6/conv1d/Squeeze╤
-sequential_12/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_12/conv1d_6/BiasAdd/ReadVariableOpш
sequential_12/conv1d_6/BiasAddBiasAdd.sequential_12/conv1d_6/conv1d/Squeeze:output:05sequential_12/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2 
sequential_12/conv1d_6/BiasAddб
sequential_12/conv1d_6/ReluRelu'sequential_12/conv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:          2
sequential_12/conv1d_6/Reluз
,sequential_12/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2.
,sequential_12/conv1d_7/conv1d/ExpandDims/dim■
(sequential_12/conv1d_7/conv1d/ExpandDims
ExpandDims)sequential_12/conv1d_6/Relu:activations:05sequential_12/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2*
(sequential_12/conv1d_7/conv1d/ExpandDims¤
9sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBsequential_12_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02;
9sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpв
.sequential_12/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/conv1d_7/conv1d/ExpandDims_1/dimУ
*sequential_12/conv1d_7/conv1d/ExpandDims_1
ExpandDimsAsequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:07sequential_12/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2,
*sequential_12/conv1d_7/conv1d/ExpandDims_1Т
sequential_12/conv1d_7/conv1dConv2D1sequential_12/conv1d_7/conv1d/ExpandDims:output:03sequential_12/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
sequential_12/conv1d_7/conv1d╫
%sequential_12/conv1d_7/conv1d/SqueezeSqueeze&sequential_12/conv1d_7/conv1d:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

¤        2'
%sequential_12/conv1d_7/conv1d/Squeeze╤
-sequential_12/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_12/conv1d_7/BiasAdd/ReadVariableOpш
sequential_12/conv1d_7/BiasAddBiasAdd.sequential_12/conv1d_7/conv1d/Squeeze:output:05sequential_12/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @2 
sequential_12/conv1d_7/BiasAddб
sequential_12/conv1d_7/ReluRelu'sequential_12/conv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:         @2
sequential_12/conv1d_7/ReluЮ
,sequential_12/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_12/max_pooling1d_2/ExpandDims/dim■
(sequential_12/max_pooling1d_2/ExpandDims
ExpandDims)sequential_12/conv1d_7/Relu:activations:05sequential_12/max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @2*
(sequential_12/max_pooling1d_2/ExpandDims°
%sequential_12/max_pooling1d_2/MaxPoolMaxPool1sequential_12/max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2'
%sequential_12/max_pooling1d_2/MaxPool╓
%sequential_12/max_pooling1d_2/SqueezeSqueeze.sequential_12/max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
2'
%sequential_12/max_pooling1d_2/SqueezeШ
sequential_12/lstm_34/ShapeShape.sequential_12/max_pooling1d_2/Squeeze:output:0*
T0*
_output_shapes
:2
sequential_12/lstm_34/Shapeа
)sequential_12/lstm_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_34/strided_slice/stackд
+sequential_12/lstm_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_34/strided_slice/stack_1д
+sequential_12/lstm_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_34/strided_slice/stack_2ц
#sequential_12/lstm_34/strided_sliceStridedSlice$sequential_12/lstm_34/Shape:output:02sequential_12/lstm_34/strided_slice/stack:output:04sequential_12/lstm_34/strided_slice/stack_1:output:04sequential_12/lstm_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_34/strided_sliceИ
!sequential_12/lstm_34/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2#
!sequential_12/lstm_34/zeros/mul/y─
sequential_12/lstm_34/zeros/mulMul,sequential_12/lstm_34/strided_slice:output:0*sequential_12/lstm_34/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_34/zeros/mulЛ
"sequential_12/lstm_34/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_12/lstm_34/zeros/Less/y┐
 sequential_12/lstm_34/zeros/LessLess#sequential_12/lstm_34/zeros/mul:z:0+sequential_12/lstm_34/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_34/zeros/LessО
$sequential_12/lstm_34/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2&
$sequential_12/lstm_34/zeros/packed/1█
"sequential_12/lstm_34/zeros/packedPack,sequential_12/lstm_34/strided_slice:output:0-sequential_12/lstm_34/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_34/zeros/packedЛ
!sequential_12/lstm_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_34/zeros/Const═
sequential_12/lstm_34/zerosFill+sequential_12/lstm_34/zeros/packed:output:0*sequential_12/lstm_34/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
sequential_12/lstm_34/zerosМ
#sequential_12/lstm_34/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2%
#sequential_12/lstm_34/zeros_1/mul/y╩
!sequential_12/lstm_34/zeros_1/mulMul,sequential_12/lstm_34/strided_slice:output:0,sequential_12/lstm_34/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_34/zeros_1/mulП
$sequential_12/lstm_34/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_12/lstm_34/zeros_1/Less/y╟
"sequential_12/lstm_34/zeros_1/LessLess%sequential_12/lstm_34/zeros_1/mul:z:0-sequential_12/lstm_34/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_34/zeros_1/LessТ
&sequential_12/lstm_34/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2(
&sequential_12/lstm_34/zeros_1/packed/1с
$sequential_12/lstm_34/zeros_1/packedPack,sequential_12/lstm_34/strided_slice:output:0/sequential_12/lstm_34/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_34/zeros_1/packedП
#sequential_12/lstm_34/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_34/zeros_1/Const╒
sequential_12/lstm_34/zeros_1Fill-sequential_12/lstm_34/zeros_1/packed:output:0,sequential_12/lstm_34/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
sequential_12/lstm_34/zeros_1б
$sequential_12/lstm_34/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_34/transpose/permф
sequential_12/lstm_34/transpose	Transpose.sequential_12/max_pooling1d_2/Squeeze:output:0-sequential_12/lstm_34/transpose/perm:output:0*
T0*+
_output_shapes
:         @2!
sequential_12/lstm_34/transposeС
sequential_12/lstm_34/Shape_1Shape#sequential_12/lstm_34/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_34/Shape_1д
+sequential_12/lstm_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_34/strided_slice_1/stackи
-sequential_12/lstm_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_34/strided_slice_1/stack_1и
-sequential_12/lstm_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_34/strided_slice_1/stack_2Є
%sequential_12/lstm_34/strided_slice_1StridedSlice&sequential_12/lstm_34/Shape_1:output:04sequential_12/lstm_34/strided_slice_1/stack:output:06sequential_12/lstm_34/strided_slice_1/stack_1:output:06sequential_12/lstm_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_34/strided_slice_1▒
1sequential_12/lstm_34/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_12/lstm_34/TensorArrayV2/element_shapeК
#sequential_12/lstm_34/TensorArrayV2TensorListReserve:sequential_12/lstm_34/TensorArrayV2/element_shape:output:0.sequential_12/lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_34/TensorArrayV2ы
Ksequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2M
Ksequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_34/transpose:y:0Tsequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensorд
+sequential_12/lstm_34/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_34/strided_slice_2/stackи
-sequential_12/lstm_34/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_34/strided_slice_2/stack_1и
-sequential_12/lstm_34/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_34/strided_slice_2/stack_2А
%sequential_12/lstm_34/strided_slice_2StridedSlice#sequential_12/lstm_34/transpose:y:04sequential_12/lstm_34/strided_slice_2/stack:output:06sequential_12/lstm_34/strided_slice_2/stack_1:output:06sequential_12/lstm_34/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2'
%sequential_12/lstm_34/strided_slice_2Ў
8sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_34_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02:
8sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOpД
)sequential_12/lstm_34/lstm_cell_34/MatMulMatMul.sequential_12/lstm_34/strided_slice_2:output:0@sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2+
)sequential_12/lstm_34/lstm_cell_34/MatMul№
:sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_34_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02<
:sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOpА
+sequential_12/lstm_34/lstm_cell_34/MatMul_1MatMul$sequential_12/lstm_34/zeros:output:0Bsequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2-
+sequential_12/lstm_34/lstm_cell_34/MatMul_1ў
&sequential_12/lstm_34/lstm_cell_34/addAddV23sequential_12/lstm_34/lstm_cell_34/MatMul:product:05sequential_12/lstm_34/lstm_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:         $2(
&sequential_12/lstm_34/lstm_cell_34/addї
9sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02;
9sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOpД
*sequential_12/lstm_34/lstm_cell_34/BiasAddBiasAdd*sequential_12/lstm_34/lstm_cell_34/add:z:0Asequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2,
*sequential_12/lstm_34/lstm_cell_34/BiasAddк
2sequential_12/lstm_34/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_34/lstm_cell_34/split/split_dim╦
(sequential_12/lstm_34/lstm_cell_34/splitSplit;sequential_12/lstm_34/lstm_cell_34/split/split_dim:output:03sequential_12/lstm_34/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2*
(sequential_12/lstm_34/lstm_cell_34/split╚
*sequential_12/lstm_34/lstm_cell_34/SigmoidSigmoid1sequential_12/lstm_34/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:         	2,
*sequential_12/lstm_34/lstm_cell_34/Sigmoid╠
,sequential_12/lstm_34/lstm_cell_34/Sigmoid_1Sigmoid1sequential_12/lstm_34/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_34/lstm_cell_34/Sigmoid_1у
&sequential_12/lstm_34/lstm_cell_34/mulMul0sequential_12/lstm_34/lstm_cell_34/Sigmoid_1:y:0&sequential_12/lstm_34/zeros_1:output:0*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_34/lstm_cell_34/mul┐
'sequential_12/lstm_34/lstm_cell_34/ReluRelu1sequential_12/lstm_34/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:         	2)
'sequential_12/lstm_34/lstm_cell_34/ReluЇ
(sequential_12/lstm_34/lstm_cell_34/mul_1Mul.sequential_12/lstm_34/lstm_cell_34/Sigmoid:y:05sequential_12/lstm_34/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_34/lstm_cell_34/mul_1щ
(sequential_12/lstm_34/lstm_cell_34/add_1AddV2*sequential_12/lstm_34/lstm_cell_34/mul:z:0,sequential_12/lstm_34/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_34/lstm_cell_34/add_1╠
,sequential_12/lstm_34/lstm_cell_34/Sigmoid_2Sigmoid1sequential_12/lstm_34/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_34/lstm_cell_34/Sigmoid_2╛
)sequential_12/lstm_34/lstm_cell_34/Relu_1Relu,sequential_12/lstm_34/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:         	2+
)sequential_12/lstm_34/lstm_cell_34/Relu_1°
(sequential_12/lstm_34/lstm_cell_34/mul_2Mul0sequential_12/lstm_34/lstm_cell_34/Sigmoid_2:y:07sequential_12/lstm_34/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_34/lstm_cell_34/mul_2╗
3sequential_12/lstm_34/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   25
3sequential_12/lstm_34/TensorArrayV2_1/element_shapeР
%sequential_12/lstm_34/TensorArrayV2_1TensorListReserve<sequential_12/lstm_34/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_34/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_34/TensorArrayV2_1z
sequential_12/lstm_34/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_34/timeл
.sequential_12/lstm_34/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_12/lstm_34/while/maximum_iterationsЦ
(sequential_12/lstm_34/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_34/while/loop_counter╒
sequential_12/lstm_34/whileWhile1sequential_12/lstm_34/while/loop_counter:output:07sequential_12/lstm_34/while/maximum_iterations:output:0#sequential_12/lstm_34/time:output:0.sequential_12/lstm_34/TensorArrayV2_1:handle:0$sequential_12/lstm_34/zeros:output:0&sequential_12/lstm_34/zeros_1:output:0.sequential_12/lstm_34/strided_slice_1:output:0Msequential_12/lstm_34/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_34_lstm_cell_34_matmul_readvariableop_resourceCsequential_12_lstm_34_lstm_cell_34_matmul_1_readvariableop_resourceBsequential_12_lstm_34_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_12_lstm_34_while_body_318947*3
cond+R)
'sequential_12_lstm_34_while_cond_318946*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
sequential_12/lstm_34/whileс
Fsequential_12/lstm_34/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2H
Fsequential_12/lstm_34/TensorArrayV2Stack/TensorListStack/element_shape└
8sequential_12/lstm_34/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_34/while:output:3Osequential_12/lstm_34/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02:
8sequential_12/lstm_34/TensorArrayV2Stack/TensorListStackн
+sequential_12/lstm_34/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_12/lstm_34/strided_slice_3/stackи
-sequential_12/lstm_34/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_34/strided_slice_3/stack_1и
-sequential_12/lstm_34/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_34/strided_slice_3/stack_2Ю
%sequential_12/lstm_34/strided_slice_3StridedSliceAsequential_12/lstm_34/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_34/strided_slice_3/stack:output:06sequential_12/lstm_34/strided_slice_3/stack_1:output:06sequential_12/lstm_34/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2'
%sequential_12/lstm_34/strided_slice_3е
&sequential_12/lstm_34/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_34/transpose_1/perm¤
!sequential_12/lstm_34/transpose_1	TransposeAsequential_12/lstm_34/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_34/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2#
!sequential_12/lstm_34/transpose_1Т
sequential_12/lstm_34/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_34/runtimeп
!sequential_12/dropout_18/IdentityIdentity%sequential_12/lstm_34/transpose_1:y:0*
T0*+
_output_shapes
:         	2#
!sequential_12/dropout_18/IdentityФ
sequential_12/lstm_35/ShapeShape*sequential_12/dropout_18/Identity:output:0*
T0*
_output_shapes
:2
sequential_12/lstm_35/Shapeа
)sequential_12/lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_35/strided_slice/stackд
+sequential_12/lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_35/strided_slice/stack_1д
+sequential_12/lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_35/strided_slice/stack_2ц
#sequential_12/lstm_35/strided_sliceStridedSlice$sequential_12/lstm_35/Shape:output:02sequential_12/lstm_35/strided_slice/stack:output:04sequential_12/lstm_35/strided_slice/stack_1:output:04sequential_12/lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_35/strided_sliceИ
!sequential_12/lstm_35/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2#
!sequential_12/lstm_35/zeros/mul/y─
sequential_12/lstm_35/zeros/mulMul,sequential_12/lstm_35/strided_slice:output:0*sequential_12/lstm_35/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_35/zeros/mulЛ
"sequential_12/lstm_35/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_12/lstm_35/zeros/Less/y┐
 sequential_12/lstm_35/zeros/LessLess#sequential_12/lstm_35/zeros/mul:z:0+sequential_12/lstm_35/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_35/zeros/LessО
$sequential_12/lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2&
$sequential_12/lstm_35/zeros/packed/1█
"sequential_12/lstm_35/zeros/packedPack,sequential_12/lstm_35/strided_slice:output:0-sequential_12/lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_35/zeros/packedЛ
!sequential_12/lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_35/zeros/Const═
sequential_12/lstm_35/zerosFill+sequential_12/lstm_35/zeros/packed:output:0*sequential_12/lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:         	2
sequential_12/lstm_35/zerosМ
#sequential_12/lstm_35/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :	2%
#sequential_12/lstm_35/zeros_1/mul/y╩
!sequential_12/lstm_35/zeros_1/mulMul,sequential_12/lstm_35/strided_slice:output:0,sequential_12/lstm_35/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_35/zeros_1/mulП
$sequential_12/lstm_35/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_12/lstm_35/zeros_1/Less/y╟
"sequential_12/lstm_35/zeros_1/LessLess%sequential_12/lstm_35/zeros_1/mul:z:0-sequential_12/lstm_35/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_35/zeros_1/LessТ
&sequential_12/lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :	2(
&sequential_12/lstm_35/zeros_1/packed/1с
$sequential_12/lstm_35/zeros_1/packedPack,sequential_12/lstm_35/strided_slice:output:0/sequential_12/lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_35/zeros_1/packedП
#sequential_12/lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_35/zeros_1/Const╒
sequential_12/lstm_35/zeros_1Fill-sequential_12/lstm_35/zeros_1/packed:output:0,sequential_12/lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:         	2
sequential_12/lstm_35/zeros_1б
$sequential_12/lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_35/transpose/permр
sequential_12/lstm_35/transpose	Transpose*sequential_12/dropout_18/Identity:output:0-sequential_12/lstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:         	2!
sequential_12/lstm_35/transposeС
sequential_12/lstm_35/Shape_1Shape#sequential_12/lstm_35/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_35/Shape_1д
+sequential_12/lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_35/strided_slice_1/stackи
-sequential_12/lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_35/strided_slice_1/stack_1и
-sequential_12/lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_35/strided_slice_1/stack_2Є
%sequential_12/lstm_35/strided_slice_1StridedSlice&sequential_12/lstm_35/Shape_1:output:04sequential_12/lstm_35/strided_slice_1/stack:output:06sequential_12/lstm_35/strided_slice_1/stack_1:output:06sequential_12/lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_35/strided_slice_1▒
1sequential_12/lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_12/lstm_35/TensorArrayV2/element_shapeК
#sequential_12/lstm_35/TensorArrayV2TensorListReserve:sequential_12/lstm_35/TensorArrayV2/element_shape:output:0.sequential_12/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_35/TensorArrayV2ы
Ksequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2M
Ksequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_35/transpose:y:0Tsequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensorд
+sequential_12/lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_35/strided_slice_2/stackи
-sequential_12/lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_35/strided_slice_2/stack_1и
-sequential_12/lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_35/strided_slice_2/stack_2А
%sequential_12/lstm_35/strided_slice_2StridedSlice#sequential_12/lstm_35/transpose:y:04sequential_12/lstm_35/strided_slice_2/stack:output:06sequential_12/lstm_35/strided_slice_2/stack_1:output:06sequential_12/lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2'
%sequential_12/lstm_35/strided_slice_2Ў
8sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype02:
8sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOpД
)sequential_12/lstm_35/lstm_cell_35/MatMulMatMul.sequential_12/lstm_35/strided_slice_2:output:0@sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2+
)sequential_12/lstm_35/lstm_cell_35/MatMul№
:sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02<
:sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpА
+sequential_12/lstm_35/lstm_cell_35/MatMul_1MatMul$sequential_12/lstm_35/zeros:output:0Bsequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2-
+sequential_12/lstm_35/lstm_cell_35/MatMul_1ў
&sequential_12/lstm_35/lstm_cell_35/addAddV23sequential_12/lstm_35/lstm_cell_35/MatMul:product:05sequential_12/lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2(
&sequential_12/lstm_35/lstm_cell_35/addї
9sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02;
9sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpД
*sequential_12/lstm_35/lstm_cell_35/BiasAddBiasAdd*sequential_12/lstm_35/lstm_cell_35/add:z:0Asequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2,
*sequential_12/lstm_35/lstm_cell_35/BiasAddк
2sequential_12/lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_35/lstm_cell_35/split/split_dim╦
(sequential_12/lstm_35/lstm_cell_35/splitSplit;sequential_12/lstm_35/lstm_cell_35/split/split_dim:output:03sequential_12/lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2*
(sequential_12/lstm_35/lstm_cell_35/split╚
*sequential_12/lstm_35/lstm_cell_35/SigmoidSigmoid1sequential_12/lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2,
*sequential_12/lstm_35/lstm_cell_35/Sigmoid╠
,sequential_12/lstm_35/lstm_cell_35/Sigmoid_1Sigmoid1sequential_12/lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_35/lstm_cell_35/Sigmoid_1у
&sequential_12/lstm_35/lstm_cell_35/mulMul0sequential_12/lstm_35/lstm_cell_35/Sigmoid_1:y:0&sequential_12/lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:         	2(
&sequential_12/lstm_35/lstm_cell_35/mul┐
'sequential_12/lstm_35/lstm_cell_35/ReluRelu1sequential_12/lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2)
'sequential_12/lstm_35/lstm_cell_35/ReluЇ
(sequential_12/lstm_35/lstm_cell_35/mul_1Mul.sequential_12/lstm_35/lstm_cell_35/Sigmoid:y:05sequential_12/lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_35/lstm_cell_35/mul_1щ
(sequential_12/lstm_35/lstm_cell_35/add_1AddV2*sequential_12/lstm_35/lstm_cell_35/mul:z:0,sequential_12/lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_35/lstm_cell_35/add_1╠
,sequential_12/lstm_35/lstm_cell_35/Sigmoid_2Sigmoid1sequential_12/lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2.
,sequential_12/lstm_35/lstm_cell_35/Sigmoid_2╛
)sequential_12/lstm_35/lstm_cell_35/Relu_1Relu,sequential_12/lstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2+
)sequential_12/lstm_35/lstm_cell_35/Relu_1°
(sequential_12/lstm_35/lstm_cell_35/mul_2Mul0sequential_12/lstm_35/lstm_cell_35/Sigmoid_2:y:07sequential_12/lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2*
(sequential_12/lstm_35/lstm_cell_35/mul_2╗
3sequential_12/lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   25
3sequential_12/lstm_35/TensorArrayV2_1/element_shapeР
%sequential_12/lstm_35/TensorArrayV2_1TensorListReserve<sequential_12/lstm_35/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_35/TensorArrayV2_1z
sequential_12/lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_35/timeл
.sequential_12/lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_12/lstm_35/while/maximum_iterationsЦ
(sequential_12/lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_35/while/loop_counter╒
sequential_12/lstm_35/whileWhile1sequential_12/lstm_35/while/loop_counter:output:07sequential_12/lstm_35/while/maximum_iterations:output:0#sequential_12/lstm_35/time:output:0.sequential_12/lstm_35/TensorArrayV2_1:handle:0$sequential_12/lstm_35/zeros:output:0&sequential_12/lstm_35/zeros_1:output:0.sequential_12/lstm_35/strided_slice_1:output:0Msequential_12/lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_35_lstm_cell_35_matmul_readvariableop_resourceCsequential_12_lstm_35_lstm_cell_35_matmul_1_readvariableop_resourceBsequential_12_lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         	:         	: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_12_lstm_35_while_body_319095*3
cond+R)
'sequential_12_lstm_35_while_cond_319094*K
output_shapes:
8: : : : :         	:         	: : : : : *
parallel_iterations 2
sequential_12/lstm_35/whileс
Fsequential_12/lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2H
Fsequential_12/lstm_35/TensorArrayV2Stack/TensorListStack/element_shape└
8sequential_12/lstm_35/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_35/while:output:3Osequential_12/lstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         	*
element_dtype02:
8sequential_12/lstm_35/TensorArrayV2Stack/TensorListStackн
+sequential_12/lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_12/lstm_35/strided_slice_3/stackи
-sequential_12/lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_35/strided_slice_3/stack_1и
-sequential_12/lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_35/strided_slice_3/stack_2Ю
%sequential_12/lstm_35/strided_slice_3StridedSliceAsequential_12/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_35/strided_slice_3/stack:output:06sequential_12/lstm_35/strided_slice_3/stack_1:output:06sequential_12/lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_mask2'
%sequential_12/lstm_35/strided_slice_3е
&sequential_12/lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_35/transpose_1/perm¤
!sequential_12/lstm_35/transpose_1	TransposeAsequential_12/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:         	2#
!sequential_12/lstm_35/transpose_1Т
sequential_12/lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_35/runtime┤
!sequential_12/dropout_19/IdentityIdentity.sequential_12/lstm_35/strided_slice_3:output:0*
T0*'
_output_shapes
:         	2#
!sequential_12/dropout_19/Identity╥
,sequential_12/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_34_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02.
,sequential_12/dense_34/MatMul/ReadVariableOp▄
sequential_12/dense_34/MatMulMatMul*sequential_12/dropout_19/Identity:output:04sequential_12/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential_12/dense_34/MatMul╤
-sequential_12/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_12/dense_34/BiasAdd/ReadVariableOp▌
sequential_12/dense_34/BiasAddBiasAdd'sequential_12/dense_34/MatMul:product:05sequential_12/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2 
sequential_12/dense_34/BiasAddЭ
sequential_12/dense_34/ReluRelu'sequential_12/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
sequential_12/dense_34/Relu╥
,sequential_12/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_35_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02.
,sequential_12/dense_35/MatMul/ReadVariableOp█
sequential_12/dense_35/MatMulMatMul)sequential_12/dense_34/Relu:activations:04sequential_12/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_12/dense_35/MatMul╤
-sequential_12/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_35/BiasAdd/ReadVariableOp▌
sequential_12/dense_35/BiasAddBiasAdd'sequential_12/dense_35/MatMul:product:05sequential_12/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_12/dense_35/BiasAddЧ
sequential_12/reshape_17/ShapeShape'sequential_12/dense_35/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_12/reshape_17/Shapeж
,sequential_12/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_12/reshape_17/strided_slice/stackк
.sequential_12/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_12/reshape_17/strided_slice/stack_1к
.sequential_12/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_12/reshape_17/strided_slice/stack_2°
&sequential_12/reshape_17/strided_sliceStridedSlice'sequential_12/reshape_17/Shape:output:05sequential_12/reshape_17/strided_slice/stack:output:07sequential_12/reshape_17/strided_slice/stack_1:output:07sequential_12/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_12/reshape_17/strided_sliceЦ
(sequential_12/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_12/reshape_17/Reshape/shape/1Ц
(sequential_12/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_12/reshape_17/Reshape/shape/2Э
&sequential_12/reshape_17/Reshape/shapePack/sequential_12/reshape_17/strided_slice:output:01sequential_12/reshape_17/Reshape/shape/1:output:01sequential_12/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/reshape_17/Reshape/shape▀
 sequential_12/reshape_17/ReshapeReshape'sequential_12/dense_35/BiasAdd:output:0/sequential_12/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:         2"
 sequential_12/reshape_17/ReshapeИ
IdentityIdentity)sequential_12/reshape_17/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         2

IdentityИ
NoOpNoOp.^sequential_12/conv1d_6/BiasAdd/ReadVariableOp:^sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp.^sequential_12/conv1d_7/BiasAdd/ReadVariableOp:^sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp.^sequential_12/dense_34/BiasAdd/ReadVariableOp-^sequential_12/dense_34/MatMul/ReadVariableOp.^sequential_12/dense_35/BiasAdd/ReadVariableOp-^sequential_12/dense_35/MatMul/ReadVariableOp:^sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp9^sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOp;^sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp^sequential_12/lstm_34/while:^sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp9^sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOp;^sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp^sequential_12/lstm_35/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 2^
-sequential_12/conv1d_6/BiasAdd/ReadVariableOp-sequential_12/conv1d_6/BiasAdd/ReadVariableOp2v
9sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp9sequential_12/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2^
-sequential_12/conv1d_7/BiasAdd/ReadVariableOp-sequential_12/conv1d_7/BiasAdd/ReadVariableOp2v
9sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp9sequential_12/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2^
-sequential_12/dense_34/BiasAdd/ReadVariableOp-sequential_12/dense_34/BiasAdd/ReadVariableOp2\
,sequential_12/dense_34/MatMul/ReadVariableOp,sequential_12/dense_34/MatMul/ReadVariableOp2^
-sequential_12/dense_35/BiasAdd/ReadVariableOp-sequential_12/dense_35/BiasAdd/ReadVariableOp2\
,sequential_12/dense_35/MatMul/ReadVariableOp,sequential_12/dense_35/MatMul/ReadVariableOp2v
9sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp9sequential_12/lstm_34/lstm_cell_34/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOp8sequential_12/lstm_34/lstm_cell_34/MatMul/ReadVariableOp2x
:sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp:sequential_12/lstm_34/lstm_cell_34/MatMul_1/ReadVariableOp2:
sequential_12/lstm_34/whilesequential_12/lstm_34/while2v
9sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp9sequential_12/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOp8sequential_12/lstm_35/lstm_cell_35/MatMul/ReadVariableOp2x
:sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:sequential_12/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2:
sequential_12/lstm_35/whilesequential_12/lstm_35/while:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
н
Ў
.__inference_sequential_12_layer_call_fn_320960
conv1d_6_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@$
	unknown_4:	$
	unknown_5:$
	unknown_6:	$
	unknown_7:	$
	unknown_8:$
	unknown_9:		

unknown_10:	

unknown_11:	

unknown_12:
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallconv1d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3209292
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_6_input
Л?
╩
while_body_321089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
Е
Ъ
)__inference_conv1d_6_layer_call_fn_322458

inputs
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3205132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ё
Г
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324128

inputs
states_0
states_10
matmul_readvariableop_resource:	$2
 matmul_1_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
MatMulУ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	$*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         $2
addМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2	
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
L:         	:         	:         	:         	*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         	2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         	2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         	2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         	2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         	2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         	2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         	2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         	2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         	2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         	2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         	2

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
?:         	:         	:         	: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         	
 
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
Х
ю
.__inference_sequential_12_layer_call_fn_321702

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@$
	unknown_4:	$
	unknown_5:$
	unknown_6:	$
	unknown_7:	$
	unknown_8:$
	unknown_9:		

unknown_10:	

unknown_11:	

unknown_12:
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3209292
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
В
ї
D__inference_dense_34_layer_call_and_return_conditional_losses_320891

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╞

у
lstm_35_while_cond_322334,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3.
*lstm_35_while_less_lstm_35_strided_slice_1D
@lstm_35_while_lstm_35_while_cond_322334___redundant_placeholder0D
@lstm_35_while_lstm_35_while_cond_322334___redundant_placeholder1D
@lstm_35_while_lstm_35_while_cond_322334___redundant_placeholder2D
@lstm_35_while_lstm_35_while_cond_322334___redundant_placeholder3
lstm_35_while_identity
Ш
lstm_35/while/LessLesslstm_35_while_placeholder*lstm_35_while_less_lstm_35_strided_slice_1*
T0*
_output_shapes
: 2
lstm_35/while/Lessu
lstm_35/while/IdentityIdentitylstm_35/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_35/while/Identity"9
lstm_35_while_identitylstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
└
G
+__inference_dropout_19_layer_call_fn_323853

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_3208782
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
╓
┤
(__inference_lstm_34_layer_call_fn_322547
inputs_0
unknown:@$
	unknown_0:	$
	unknown_1:$
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_34_layer_call_and_return_conditional_losses_3195982
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  	2

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
└J
╩

lstm_35_while_body_322335,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3+
'lstm_35_while_lstm_35_strided_slice_1_0g
clstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	$O
=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$J
<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:$
lstm_35_while_identity
lstm_35_while_identity_1
lstm_35_while_identity_2
lstm_35_while_identity_3
lstm_35_while_identity_4
lstm_35_while_identity_5)
%lstm_35_while_lstm_35_strided_slice_1e
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorK
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	$M
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	$H
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:$Ив1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpв0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpв2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp╙
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   2A
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0lstm_35_while_placeholderHlstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype023
1lstm_35/while/TensorArrayV2Read/TensorListGetItemр
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype022
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpЎ
!lstm_35/while/lstm_cell_35/MatMulMatMul8lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2#
!lstm_35/while/lstm_cell_35/MatMulц
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype024
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp▀
#lstm_35/while/lstm_cell_35/MatMul_1MatMullstm_35_while_placeholder_2:lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2%
#lstm_35/while/lstm_cell_35/MatMul_1╫
lstm_35/while/lstm_cell_35/addAddV2+lstm_35/while/lstm_cell_35/MatMul:product:0-lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2 
lstm_35/while/lstm_cell_35/add▀
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype023
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpф
"lstm_35/while/lstm_cell_35/BiasAddBiasAdd"lstm_35/while/lstm_cell_35/add:z:09lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2$
"lstm_35/while/lstm_cell_35/BiasAddЪ
*lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_35/while/lstm_cell_35/split/split_dimл
 lstm_35/while/lstm_cell_35/splitSplit3lstm_35/while/lstm_cell_35/split/split_dim:output:0+lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2"
 lstm_35/while/lstm_cell_35/split░
"lstm_35/while/lstm_cell_35/SigmoidSigmoid)lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2$
"lstm_35/while/lstm_cell_35/Sigmoid┤
$lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid)lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2&
$lstm_35/while/lstm_cell_35/Sigmoid_1└
lstm_35/while/lstm_cell_35/mulMul(lstm_35/while/lstm_cell_35/Sigmoid_1:y:0lstm_35_while_placeholder_3*
T0*'
_output_shapes
:         	2 
lstm_35/while/lstm_cell_35/mulз
lstm_35/while/lstm_cell_35/ReluRelu)lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2!
lstm_35/while/lstm_cell_35/Relu╘
 lstm_35/while/lstm_cell_35/mul_1Mul&lstm_35/while/lstm_cell_35/Sigmoid:y:0-lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/mul_1╔
 lstm_35/while/lstm_cell_35/add_1AddV2"lstm_35/while/lstm_cell_35/mul:z:0$lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/add_1┤
$lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid)lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2&
$lstm_35/while/lstm_cell_35/Sigmoid_2ж
!lstm_35/while/lstm_cell_35/Relu_1Relu$lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2#
!lstm_35/while/lstm_cell_35/Relu_1╪
 lstm_35/while/lstm_cell_35/mul_2Mul(lstm_35/while/lstm_cell_35/Sigmoid_2:y:0/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2"
 lstm_35/while/lstm_cell_35/mul_2И
2lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_35_while_placeholder_1lstm_35_while_placeholder$lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_35/while/TensorArrayV2Write/TensorListSetIteml
lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_35/while/add/yЙ
lstm_35/while/addAddV2lstm_35_while_placeholderlstm_35/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_35/while/addp
lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_35/while/add_1/yЮ
lstm_35/while/add_1AddV2(lstm_35_while_lstm_35_while_loop_counterlstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_35/while/add_1Л
lstm_35/while/IdentityIdentitylstm_35/while/add_1:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identityж
lstm_35/while/Identity_1Identity.lstm_35_while_lstm_35_while_maximum_iterations^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_1Н
lstm_35/while/Identity_2Identitylstm_35/while/add:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_2║
lstm_35/while/Identity_3IdentityBlstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_35/while/NoOp*
T0*
_output_shapes
: 2
lstm_35/while/Identity_3н
lstm_35/while/Identity_4Identity$lstm_35/while/lstm_cell_35/mul_2:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_35/while/Identity_4н
lstm_35/while/Identity_5Identity$lstm_35/while/lstm_cell_35/add_1:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:         	2
lstm_35/while/Identity_5Ж
lstm_35/while/NoOpNoOp2^lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1^lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp3^lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_35/while/NoOp"9
lstm_35_while_identitylstm_35/while/Identity:output:0"=
lstm_35_while_identity_1!lstm_35/while/Identity_1:output:0"=
lstm_35_while_identity_2!lstm_35/while/Identity_2:output:0"=
lstm_35_while_identity_3!lstm_35/while/Identity_3:output:0"=
lstm_35_while_identity_4!lstm_35/while/Identity_4:output:0"=
lstm_35_while_identity_5!lstm_35/while/Identity_5:output:0"P
%lstm_35_while_lstm_35_strided_slice_1'lstm_35_while_lstm_35_strided_slice_1_0"z
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"|
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"x
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"╚
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2f
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2d
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2h
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: 
╒
├
while_cond_322635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322635___redundant_placeholder04
0while_while_cond_322635___redundant_placeholder14
0while_while_cond_322635___redundant_placeholder24
0while_while_cond_322635___redundant_placeholder3
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
@: : : : :         	:         	: ::::: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
:
Л?
╩
while_body_323311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_35_matmul_readvariableop_resource_0:	$G
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	$B
4while_lstm_cell_35_biasadd_readvariableop_resource_0:$
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_35_matmul_readvariableop_resource:	$E
3while_lstm_cell_35_matmul_1_readvariableop_resource:	$@
2while_lstm_cell_35_biasadd_readvariableop_resource:$Ив)while/lstm_cell_35/BiasAdd/ReadVariableOpв(while/lstm_cell_35/MatMul/ReadVariableOpв*while/lstm_cell_35/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╚
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:	$*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp╓
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul╬
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:	$*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp┐
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/MatMul_1╖
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/add╟
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:$*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp─
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         $2
while/lstm_cell_35/BiasAddК
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dimЛ
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:         	:         	:         	:         	*
	num_split2
while/lstm_cell_35/splitШ
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/SigmoidЬ
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_1а
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mulП
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu┤
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_1й
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/add_1Ь
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Sigmoid_2О
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/Relu_1╕
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:         	2
while/lstm_cell_35/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         	2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         	:         	: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 
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
:         	:-)
'
_output_shapes
:         	:

_output_shapes
: :

_output_shapes
: "иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_defaultп
M
conv1d_6_input;
 serving_default_conv1d_6_input:0         B

reshape_174
StatefulPartitionedCall:0         tensorflow/serving/predict:┴А
З
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
╛__call__
+┐&call_and_return_all_conditional_losses
└_default_save_signature"
_tf_keras_sequential
╜

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
├__call__
+─&call_and_return_all_conditional_losses"
_tf_keras_layer
з
regularization_losses
trainable_variables
	variables
 	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
!cell
"
state_spec
#regularization_losses
$trainable_variables
%	variables
&	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
'regularization_losses
(trainable_variables
)	variables
*	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
+cell
,
state_spec
-regularization_losses
.trainable_variables
/	variables
0	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
1regularization_losses
2trainable_variables
3	variables
4	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"
_tf_keras_layer
з
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratemвmгmдmе5mж6mз;mи<mйJmкKmлLmмMmнNmоOmпv░v▒v▓v│5v┤6v╡;v╢<v╖Jv╕Kv╣Lv║Mv╗Nv╝Ov╜"
	optimizer
 "
trackable_list_wrapper
Ж
0
1
2
3
J4
K5
L6
M7
N8
O9
510
611
;12
<13"
trackable_list_wrapper
Ж
0
1
2
3
J4
K5
L6
M7
N8
O9
510
611
;12
<13"
trackable_list_wrapper
╬
Player_regularization_losses
regularization_losses
Qlayer_metrics
trainable_variables

Rlayers
Snon_trainable_variables
Tmetrics
	variables
╛__call__
└_default_save_signature
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
-
╒serving_default"
signature_map
%:# 2conv1d_6/kernel
: 2conv1d_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Ulayer_regularization_losses
regularization_losses
Vlayer_metrics
trainable_variables

Wlayers
Xnon_trainable_variables
Ymetrics
	variables
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
%:# @2conv1d_7/kernel
:@2conv1d_7/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Zlayer_regularization_losses
regularization_losses
[layer_metrics
trainable_variables

\layers
]non_trainable_variables
^metrics
	variables
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
_layer_regularization_losses
regularization_losses
`layer_metrics
trainable_variables

alayers
bnon_trainable_variables
cmetrics
	variables
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
у
d
state_size

Jkernel
Krecurrent_kernel
Lbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
╝
ilayer_regularization_losses
#regularization_losses
jlayer_metrics
$trainable_variables

klayers

lstates
mnon_trainable_variables
nmetrics
%	variables
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
olayer_regularization_losses
'regularization_losses
player_metrics
(trainable_variables

qlayers
rnon_trainable_variables
smetrics
)	variables
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
у
t
state_size

Mkernel
Nrecurrent_kernel
Obias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
╪__call__
+┘&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
╝
ylayer_regularization_losses
-regularization_losses
zlayer_metrics
.trainable_variables

{layers

|states
}non_trainable_variables
~metrics
/	variables
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
layer_regularization_losses
1regularization_losses
Аlayer_metrics
2trainable_variables
Бlayers
Вnon_trainable_variables
Гmetrics
3	variables
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
!:		2dense_34/kernel
:	2dense_34/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
╡
 Дlayer_regularization_losses
7regularization_losses
Еlayer_metrics
8trainable_variables
Жlayers
Зnon_trainable_variables
Иmetrics
9	variables
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_35/kernel
:2dense_35/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
╡
 Йlayer_regularization_losses
=regularization_losses
Кlayer_metrics
>trainable_variables
Лlayers
Мnon_trainable_variables
Нmetrics
?	variables
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Оlayer_regularization_losses
Aregularization_losses
Пlayer_metrics
Btrainable_variables
Рlayers
Сnon_trainable_variables
Тmetrics
C	variables
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@$2lstm_34/lstm_cell_34/kernel
7:5	$2%lstm_34/lstm_cell_34/recurrent_kernel
':%$2lstm_34/lstm_cell_34/bias
-:+	$2lstm_35/lstm_cell_35/kernel
7:5	$2%lstm_35/lstm_cell_35/recurrent_kernel
':%$2lstm_35/lstm_cell_35/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
(
У0"
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
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
╡
 Фlayer_regularization_losses
eregularization_losses
Хlayer_metrics
ftrainable_variables
Цlayers
Чnon_trainable_variables
Шmetrics
g	variables
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
!0"
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
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
╡
 Щlayer_regularization_losses
uregularization_losses
Ъlayer_metrics
vtrainable_variables
Ыlayers
Ьnon_trainable_variables
Эmetrics
w	variables
╪__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
+0"
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
R

Юtotal

Яcount
а	variables
б	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
Ю0
Я1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
*:( 2Adam/conv1d_6/kernel/m
 : 2Adam/conv1d_6/bias/m
*:( @2Adam/conv1d_7/kernel/m
 :@2Adam/conv1d_7/bias/m
&:$		2Adam/dense_34/kernel/m
 :	2Adam/dense_34/bias/m
&:$	2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
2:0@$2"Adam/lstm_34/lstm_cell_34/kernel/m
<::	$2,Adam/lstm_34/lstm_cell_34/recurrent_kernel/m
,:*$2 Adam/lstm_34/lstm_cell_34/bias/m
2:0	$2"Adam/lstm_35/lstm_cell_35/kernel/m
<::	$2,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m
,:*$2 Adam/lstm_35/lstm_cell_35/bias/m
*:( 2Adam/conv1d_6/kernel/v
 : 2Adam/conv1d_6/bias/v
*:( @2Adam/conv1d_7/kernel/v
 :@2Adam/conv1d_7/bias/v
&:$		2Adam/dense_34/kernel/v
 :	2Adam/dense_34/bias/v
&:$	2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v
2:0@$2"Adam/lstm_34/lstm_cell_34/kernel/v
<::	$2,Adam/lstm_34/lstm_cell_34/recurrent_kernel/v
,:*$2 Adam/lstm_34/lstm_cell_34/bias/v
2:0	$2"Adam/lstm_35/lstm_cell_35/kernel/v
<::	$2,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v
,:*$2 Adam/lstm_35/lstm_cell_35/bias/v
Ж2Г
.__inference_sequential_12_layer_call_fn_320960
.__inference_sequential_12_layer_call_fn_321702
.__inference_sequential_12_layer_call_fn_321735
.__inference_sequential_12_layer_call_fn_321544└
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
Є2я
I__inference_sequential_12_layer_call_and_return_conditional_losses_322085
I__inference_sequential_12_layer_call_and_return_conditional_losses_322449
I__inference_sequential_12_layer_call_and_return_conditional_losses_321586
I__inference_sequential_12_layer_call_and_return_conditional_losses_321628└
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
╙B╨
!__inference__wrapped_model_319202conv1d_6_input"Ш
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
╙2╨
)__inference_conv1d_6_layer_call_fn_322458в
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
ю2ы
D__inference_conv1d_6_layer_call_and_return_conditional_losses_322474в
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
╙2╨
)__inference_conv1d_7_layer_call_fn_322483в
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
ю2ы
D__inference_conv1d_7_layer_call_and_return_conditional_losses_322499в
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
М2Й
0__inference_max_pooling1d_2_layer_call_fn_322504
0__inference_max_pooling1d_2_layer_call_fn_322509в
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
┬2┐
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322517
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322525в
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
Г2А
(__inference_lstm_34_layer_call_fn_322536
(__inference_lstm_34_layer_call_fn_322547
(__inference_lstm_34_layer_call_fn_322558
(__inference_lstm_34_layer_call_fn_322569╒
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
я2ь
C__inference_lstm_34_layer_call_and_return_conditional_losses_322720
C__inference_lstm_34_layer_call_and_return_conditional_losses_322871
C__inference_lstm_34_layer_call_and_return_conditional_losses_323022
C__inference_lstm_34_layer_call_and_return_conditional_losses_323173╒
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
Ф2С
+__inference_dropout_18_layer_call_fn_323178
+__inference_dropout_18_layer_call_fn_323183┤
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
╩2╟
F__inference_dropout_18_layer_call_and_return_conditional_losses_323188
F__inference_dropout_18_layer_call_and_return_conditional_losses_323200┤
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
Г2А
(__inference_lstm_35_layer_call_fn_323211
(__inference_lstm_35_layer_call_fn_323222
(__inference_lstm_35_layer_call_fn_323233
(__inference_lstm_35_layer_call_fn_323244╒
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
я2ь
C__inference_lstm_35_layer_call_and_return_conditional_losses_323395
C__inference_lstm_35_layer_call_and_return_conditional_losses_323546
C__inference_lstm_35_layer_call_and_return_conditional_losses_323697
C__inference_lstm_35_layer_call_and_return_conditional_losses_323848╒
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
Ф2С
+__inference_dropout_19_layer_call_fn_323853
+__inference_dropout_19_layer_call_fn_323858┤
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
╩2╟
F__inference_dropout_19_layer_call_and_return_conditional_losses_323863
F__inference_dropout_19_layer_call_and_return_conditional_losses_323875┤
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
╙2╨
)__inference_dense_34_layer_call_fn_323884в
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
ю2ы
D__inference_dense_34_layer_call_and_return_conditional_losses_323895в
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
╙2╨
)__inference_dense_35_layer_call_fn_323904в
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
ю2ы
D__inference_dense_35_layer_call_and_return_conditional_losses_323914в
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
╒2╥
+__inference_reshape_17_layer_call_fn_323919в
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
Ё2э
F__inference_reshape_17_layer_call_and_return_conditional_losses_323932в
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
╥B╧
$__inference_signature_wrapper_321669conv1d_6_input"Ф
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
в2Я
-__inference_lstm_cell_34_layer_call_fn_323949
-__inference_lstm_cell_34_layer_call_fn_323966╛
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
╪2╒
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_323998
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_324030╛
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
в2Я
-__inference_lstm_cell_35_layer_call_fn_324047
-__inference_lstm_cell_35_layer_call_fn_324064╛
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
╪2╒
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324096
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324128╛
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
 ░
!__inference__wrapped_model_319202КJKLMNO56;<;в8
1в.
,К)
conv1d_6_input         
к ";к8
6

reshape_17(К%

reshape_17         м
D__inference_conv1d_6_layer_call_and_return_conditional_losses_322474d3в0
)в&
$К!
inputs         
к ")в&
К
0          
Ъ Д
)__inference_conv1d_6_layer_call_fn_322458W3в0
)в&
$К!
inputs         
к "К          м
D__inference_conv1d_7_layer_call_and_return_conditional_losses_322499d3в0
)в&
$К!
inputs          
к ")в&
К
0         @
Ъ Д
)__inference_conv1d_7_layer_call_fn_322483W3в0
)в&
$К!
inputs          
к "К         @д
D__inference_dense_34_layer_call_and_return_conditional_losses_323895\56/в,
%в"
 К
inputs         	
к "%в"
К
0         	
Ъ |
)__inference_dense_34_layer_call_fn_323884O56/в,
%в"
 К
inputs         	
к "К         	д
D__inference_dense_35_layer_call_and_return_conditional_losses_323914\;</в,
%в"
 К
inputs         	
к "%в"
К
0         
Ъ |
)__inference_dense_35_layer_call_fn_323904O;</в,
%в"
 К
inputs         	
к "К         о
F__inference_dropout_18_layer_call_and_return_conditional_losses_323188d7в4
-в*
$К!
inputs         	
p 
к ")в&
К
0         	
Ъ о
F__inference_dropout_18_layer_call_and_return_conditional_losses_323200d7в4
-в*
$К!
inputs         	
p
к ")в&
К
0         	
Ъ Ж
+__inference_dropout_18_layer_call_fn_323178W7в4
-в*
$К!
inputs         	
p 
к "К         	Ж
+__inference_dropout_18_layer_call_fn_323183W7в4
-в*
$К!
inputs         	
p
к "К         	ж
F__inference_dropout_19_layer_call_and_return_conditional_losses_323863\3в0
)в&
 К
inputs         	
p 
к "%в"
К
0         	
Ъ ж
F__inference_dropout_19_layer_call_and_return_conditional_losses_323875\3в0
)в&
 К
inputs         	
p
к "%в"
К
0         	
Ъ ~
+__inference_dropout_19_layer_call_fn_323853O3в0
)в&
 К
inputs         	
p 
к "К         	~
+__inference_dropout_19_layer_call_fn_323858O3в0
)в&
 К
inputs         	
p
к "К         	╥
C__inference_lstm_34_layer_call_and_return_conditional_losses_322720КJKLOвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "2в/
(К%
0                  	
Ъ ╥
C__inference_lstm_34_layer_call_and_return_conditional_losses_322871КJKLOвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "2в/
(К%
0                  	
Ъ ╕
C__inference_lstm_34_layer_call_and_return_conditional_losses_323022qJKL?в<
5в2
$К!
inputs         @

 
p 

 
к ")в&
К
0         	
Ъ ╕
C__inference_lstm_34_layer_call_and_return_conditional_losses_323173qJKL?в<
5в2
$К!
inputs         @

 
p

 
к ")в&
К
0         	
Ъ й
(__inference_lstm_34_layer_call_fn_322536}JKLOвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "%К"                  	й
(__inference_lstm_34_layer_call_fn_322547}JKLOвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "%К"                  	Р
(__inference_lstm_34_layer_call_fn_322558dJKL?в<
5в2
$К!
inputs         @

 
p 

 
к "К         	Р
(__inference_lstm_34_layer_call_fn_322569dJKL?в<
5в2
$К!
inputs         @

 
p

 
к "К         	─
C__inference_lstm_35_layer_call_and_return_conditional_losses_323395}MNOOвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p 

 
к "%в"
К
0         	
Ъ ─
C__inference_lstm_35_layer_call_and_return_conditional_losses_323546}MNOOвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p

 
к "%в"
К
0         	
Ъ ┤
C__inference_lstm_35_layer_call_and_return_conditional_losses_323697mMNO?в<
5в2
$К!
inputs         	

 
p 

 
к "%в"
К
0         	
Ъ ┤
C__inference_lstm_35_layer_call_and_return_conditional_losses_323848mMNO?в<
5в2
$К!
inputs         	

 
p

 
к "%в"
К
0         	
Ъ Ь
(__inference_lstm_35_layer_call_fn_323211pMNOOвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p 

 
к "К         	Ь
(__inference_lstm_35_layer_call_fn_323222pMNOOвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p

 
к "К         	М
(__inference_lstm_35_layer_call_fn_323233`MNO?в<
5в2
$К!
inputs         	

 
p 

 
к "К         	М
(__inference_lstm_35_layer_call_fn_323244`MNO?в<
5в2
$К!
inputs         	

 
p

 
к "К         	╩
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_323998¤JKLАв}
vвs
 К
inputs         @
KвH
"К
states/0         	
"К
states/1         	
p 
к "sвp
iвf
К
0/0         	
EЪB
К
0/1/0         	
К
0/1/1         	
Ъ ╩
H__inference_lstm_cell_34_layer_call_and_return_conditional_losses_324030¤JKLАв}
vвs
 К
inputs         @
KвH
"К
states/0         	
"К
states/1         	
p
к "sвp
iвf
К
0/0         	
EЪB
К
0/1/0         	
К
0/1/1         	
Ъ Я
-__inference_lstm_cell_34_layer_call_fn_323949эJKLАв}
vвs
 К
inputs         @
KвH
"К
states/0         	
"К
states/1         	
p 
к "cв`
К
0         	
AЪ>
К
1/0         	
К
1/1         	Я
-__inference_lstm_cell_34_layer_call_fn_323966эJKLАв}
vвs
 К
inputs         @
KвH
"К
states/0         	
"К
states/1         	
p
к "cв`
К
0         	
AЪ>
К
1/0         	
К
1/1         	╩
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324096¤MNOАв}
vвs
 К
inputs         	
KвH
"К
states/0         	
"К
states/1         	
p 
к "sвp
iвf
К
0/0         	
EЪB
К
0/1/0         	
К
0/1/1         	
Ъ ╩
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_324128¤MNOАв}
vвs
 К
inputs         	
KвH
"К
states/0         	
"К
states/1         	
p
к "sвp
iвf
К
0/0         	
EЪB
К
0/1/0         	
К
0/1/1         	
Ъ Я
-__inference_lstm_cell_35_layer_call_fn_324047эMNOАв}
vвs
 К
inputs         	
KвH
"К
states/0         	
"К
states/1         	
p 
к "cв`
К
0         	
AЪ>
К
1/0         	
К
1/1         	Я
-__inference_lstm_cell_35_layer_call_fn_324064эMNOАв}
vвs
 К
inputs         	
KвH
"К
states/0         	
"К
states/1         	
p
к "cв`
К
0         	
AЪ>
К
1/0         	
К
1/1         	╘
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322517ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ п
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_322525`3в0
)в&
$К!
inputs         @
к ")в&
К
0         @
Ъ л
0__inference_max_pooling1d_2_layer_call_fn_322504wEвB
;в8
6К3
inputs'                           
к ".К+'                           З
0__inference_max_pooling1d_2_layer_call_fn_322509S3в0
)в&
$К!
inputs         @
к "К         @ж
F__inference_reshape_17_layer_call_and_return_conditional_losses_323932\/в,
%в"
 К
inputs         
к ")в&
К
0         
Ъ ~
+__inference_reshape_17_layer_call_fn_323919O/в,
%в"
 К
inputs         
к "К         ╬
I__inference_sequential_12_layer_call_and_return_conditional_losses_321586АJKLMNO56;<Cв@
9в6
,К)
conv1d_6_input         
p 

 
к ")в&
К
0         
Ъ ╬
I__inference_sequential_12_layer_call_and_return_conditional_losses_321628АJKLMNO56;<Cв@
9в6
,К)
conv1d_6_input         
p

 
к ")в&
К
0         
Ъ ┼
I__inference_sequential_12_layer_call_and_return_conditional_losses_322085xJKLMNO56;<;в8
1в.
$К!
inputs         
p 

 
к ")в&
К
0         
Ъ ┼
I__inference_sequential_12_layer_call_and_return_conditional_losses_322449xJKLMNO56;<;в8
1в.
$К!
inputs         
p

 
к ")в&
К
0         
Ъ е
.__inference_sequential_12_layer_call_fn_320960sJKLMNO56;<Cв@
9в6
,К)
conv1d_6_input         
p 

 
к "К         е
.__inference_sequential_12_layer_call_fn_321544sJKLMNO56;<Cв@
9в6
,К)
conv1d_6_input         
p

 
к "К         Э
.__inference_sequential_12_layer_call_fn_321702kJKLMNO56;<;в8
1в.
$К!
inputs         
p 

 
к "К         Э
.__inference_sequential_12_layer_call_fn_321735kJKLMNO56;<;в8
1в.
$К!
inputs         
p

 
к "К         ┼
$__inference_signature_wrapper_321669ЬJKLMNO56;<MвJ
в 
Cк@
>
conv1d_6_input,К)
conv1d_6_input         ";к8
6

reshape_17(К%

reshape_17         